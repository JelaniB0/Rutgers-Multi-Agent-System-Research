import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from agents2.azure_openai import (
    AzureOpenAISettings,
    ModelConfig,
    create_agent_client,
    create_async_openai_client,
    get_azure_openai_settings,
)
from agents2.dag_builder import DAG_SYSTEM_PROMPT, parse_prereqs_with_llm
from agents2.data_agent import DataAgent

MODELS = ModelConfig(*(["test-deployment"] * 7))
MODEL_ENV = {f"AZURE_OPENAI_{role.upper()}_DEPLOYMENT": value for role, value in MODELS.to_dict().items()}


class AzureOpenAISettingsTests(unittest.TestCase):
    def test_each_role_loads_its_explicit_deployment(self):
        environment = {"AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/openai/v1",
                       "AZURE_OPENAI_API_KEY": "test-key"}
        expected = {role: f" deployment-{role} " for role in MODELS.to_dict()}
        environment.update({f"AZURE_OPENAI_{role.upper()}_DEPLOYMENT": value for role, value in expected.items()})
        with patch("agents2.azure_openai.load_dotenv"), patch.dict(os.environ, environment, clear=True):
            settings = get_azure_openai_settings()
        self.assertEqual(settings.models.to_dict(), {role: value.strip() for role, value in expected.items()})

    def test_each_missing_or_blank_role_fails_even_with_global_default(self):
        for variable in MODEL_ENV:
            for value in (None, "   "):
                with self.subTest(variable=variable, value=value):
                    environment = dict(MODEL_ENV, AZURE_OPENAI_ENDPOINT="https://example.openai.azure.com/openai/v1",
                                       AZURE_OPENAI_API_KEY="test-key", AZURE_OPENAI_DEPLOYMENT="global-default")
                    if value is None:
                        del environment[variable]
                    else:
                        environment[variable] = value
                    with patch("agents2.azure_openai.load_dotenv"), patch.dict(os.environ, environment, clear=True):
                        with self.assertRaisesRegex(RuntimeError, variable):
                            get_azure_openai_settings()

    def test_loads_and_normalizes_settings(self):
        environment = {
            **MODEL_ENV,
            "AZURE_OPENAI_ENDPOINT": "https://example.services.ai.azure.com/openai/v1/",
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_DEPLOYMENT": "course-gpt-5",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "course-embeddings",
        }

        with patch("agents2.azure_openai.load_dotenv"), patch.dict(
            os.environ, environment, clear=True
        ):
            settings = get_azure_openai_settings()

        self.assertEqual(settings.endpoint, "https://example.services.ai.azure.com/openai/v1")
        self.assertEqual(settings.api_key, "test-key")
        self.assertFalse(hasattr(settings, "chat_deployment"))
        self.assertEqual(settings.embedding_deployment, "course-embeddings")

    def test_rejects_project_endpoint(self):
        environment = {
            "AZURE_OPENAI_ENDPOINT": "https://example.services.ai.azure.com/api/projects/course-planner",
            "AZURE_OPENAI_API_KEY": "test-key",
        }

        with patch("agents2.azure_openai.load_dotenv"), patch.dict(
            os.environ, environment, clear=True
        ):
            with self.assertRaisesRegex(RuntimeError, "project endpoint"):
                get_azure_openai_settings()

    def test_requires_api_key(self):
        environment = {
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/openai/v1",
        }

        with patch("agents2.azure_openai.load_dotenv"), patch.dict(
            os.environ, environment, clear=True
        ):
            with self.assertRaisesRegex(RuntimeError, "AZURE_OPENAI_API_KEY"):
                get_azure_openai_settings()

    def test_requires_openai_endpoint(self):
        environment = {"AZURE_OPENAI_API_KEY": "test-key"}

        with patch("agents2.azure_openai.load_dotenv"), patch.dict(
            os.environ, environment, clear=True
        ):
            with self.assertRaisesRegex(RuntimeError, "AZURE_OPENAI_ENDPOINT"):
                get_azure_openai_settings()

    def test_settings_repr_redacts_api_key(self):
        settings = AzureOpenAISettings(
            endpoint="https://example.openai.azure.com/openai/v1",
            api_key="secret-test-key",
            embedding_deployment="course-embeddings",
            models=MODELS,
        )

        self.assertNotIn("secret-test-key", repr(settings))

    def test_client_factories_forward_azure_configuration(self):
        settings = AzureOpenAISettings(
            endpoint="https://example.openai.azure.com/openai/v1",
            api_key="test-key",
            embedding_deployment="course-embeddings",
            models=MODELS,
        )

        with patch("agents2.azure_openai.MeteredResponsesClient") as responses_client:
            create_agent_client(settings)
            responses_client.assert_called_once_with(
                base_url=settings.endpoint,
                api_key=settings.api_key,
                model_id=settings.models.orchestrator,
            )

        with patch("agents2.azure_openai.AsyncOpenAI") as async_client:
            create_async_openai_client(settings)
            async_client.assert_called_once_with(
                base_url=settings.endpoint,
                api_key=settings.api_key,
            )


class ResponsesMigrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_dag_parser_uses_responses_api(self):
        create = AsyncMock(
            return_value=SimpleNamespace(
                output_text='```json\n{"and": ["01:198:112"], "or_groups": [], '
                '"requires_permission": false}\n```'
            )
        )
        client = SimpleNamespace(responses=SimpleNamespace(create=create))

        parsed = await parse_prereqs_with_llm(
            client,
            "Prerequisite: 01:198:112.",
            model_id="course-gpt-5",
        )

        self.assertEqual(parsed["and"], ["01:198:112"])
        create.assert_awaited_once_with(
            model="course-gpt-5",
            instructions=DAG_SYSTEM_PROMPT,
            input="Parse this prerequisite text:\nPrerequisite: 01:198:112.",
            max_output_tokens=1000,
            reasoning={"effort": "minimal"},
        )

    def test_chroma_uses_embedding_deployment(self):
        settings = AzureOpenAISettings(
            endpoint="https://example.openai.azure.com/openai/v1",
            api_key="test-key",
            embedding_deployment="course-embeddings",
            models=MODELS,
        )
        collection = Mock()
        persistent_client = Mock()
        persistent_client.get_collection.return_value = collection

        with patch(
            "agents2.data_agent.get_azure_openai_settings", return_value=settings
        ), patch(
            "agents2.data_agent.chromadb.PersistentClient", return_value=persistent_client
        ), patch(
            "agents2.data_agent.embedding_functions.OpenAIEmbeddingFunction"
        ) as embedding_function:
            result = DataAgent._initialize_vector_db(SimpleNamespace(courses_data=[]))

        self.assertIs(result, collection)
        embedding_function.assert_called_once_with(
            api_key=settings.api_key,
            api_base=settings.endpoint,
            model_name=settings.embedding_deployment,
        )

    def test_latest_iteration_has_no_legacy_github_model_configuration(self):
        root = Path(__file__).resolve().parents[1]
        files = [root / "driver3.py", *sorted((root / "agents2").glob("*.py"))]
        source = "\n".join(path.read_text(encoding="utf-8") for path in files)

        for legacy_value in (
            "GITHUB_TOKEN",
            "GITHUB_ENDPOINT",
            "GITHUB_MODEL_ID",
            "OpenAIChatClient",
            "chat.completions.create",
            "gpt-4.1-mini",
        ):
            self.assertNotIn(legacy_value, source)


if __name__ == "__main__":
    unittest.main()

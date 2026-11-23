"""KBot 运维知识模型 UUID 直接用于推理。"""

from uuid import uuid4
from types import SimpleNamespace
from unittest.mock import AsyncMock
import unittest

from aiops_agent.adapters.knowledge_models import KnowledgeModelGateway
from aiops_agent.composition.knowledge_profiles import KnowledgeFoundationConfig
from knowledge_algorithms.contracts.errors import AlgorithmError
from knowledge_algorithms.contracts.wire import encode
from knowledge_algorithms.projections.contracts import ModelInput, RepresentationSpec


MODEL_ID = str(uuid4())


def spec(kind="DENSE", model_id=MODEL_ID):
    return RepresentationSpec(model_id, "checkpoint", "processor", "plain", 2, kind,
        "TEXT" if kind == "DENSE" else "IMAGE", "COSINE" if kind == "DENSE" else "DOT", "L2",
        frozenset({"TEXT_TO_TEXT"} if kind == "DENSE" else {"TEXT_TO_IMAGE", "IMAGE_TO_IMAGE"}))


class ModelUuidBindingTests(unittest.IsolatedAsyncioTestCase):
    async def test_inference_uses_catalog_uuid(self):
        for kind in ("DENSE", "MULTIVECTOR"):
            space = spec(kind)
            receipt = dict(model_revision=space.revision, processor_revision=space.processor_revision,
                input_template=space.input_template, dimension=2, normalization="L2", similarity=space.similarity,
                directions=sorted(space.directions), vector=[1., 0.], vectors=[[1., 0.]], mask=[True])
            client = SimpleNamespace(represent=AsyncMock(return_value=receipt))
            gateway = KnowledgeModelGateway(dense_client=client, visual_client=client, auth_context=None,
                image_reader=None, models={space.space_id: {"model_ref":space.model_ref}}, max_elements=100)
            await gateway.represent(ModelInput("TEXT", text="查询"), space)
            payload = client.represent.await_args.args[0]
            self.assertEqual(payload["model_id"], space.model_ref)
            self.assertNotIn("model_ref", payload)

    async def test_missing_binding_is_rejected_before_network_request(self):
        space = spec()
        client = SimpleNamespace(represent=AsyncMock())
        gateway = KnowledgeModelGateway(dense_client=client, visual_client=client, auth_context=None,
            image_reader=None, models={}, max_elements=100)
        with self.assertRaises(AlgorithmError):
            await gateway.represent(ModelInput("TEXT", text="查询"), space)
        client.represent.assert_not_awaited()

    def test_configuration_requires_model_uuid(self):
        with self.assertRaises((ValueError, AlgorithmError)):
            KnowledgeFoundationConfig(representations=[encode(spec(model_id="old-name"))]).profiles()
        space = spec()
        self.assertEqual(KnowledgeFoundationConfig(representations=[encode(space)]).profiles()[1].representations, (space,))
        self.assertEqual(KnowledgeFoundationConfig().profiles()[1].representations, ())


if __name__ == "__main__":
    unittest.main()

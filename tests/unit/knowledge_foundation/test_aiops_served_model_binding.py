"""KBot 运维知识模型 UUID 与推理服务名的显式绑定。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
import unittest

from aiops_agent.adapters.knowledge_models import KnowledgeModelGateway
from aiops_agent.composition.knowledge_profiles import KnowledgeFoundationConfig
from knowledge_algorithms.contracts.errors import AlgorithmError
from knowledge_algorithms.contracts.wire import encode
from knowledge_algorithms.projections.contracts import ModelInput, RepresentationSpec


def spec(kind="DENSE"):
    return RepresentationSpec("model-uuid", "checkpoint", "processor", "plain", 2, kind,
        "TEXT" if kind == "DENSE" else "IMAGE", "COSINE" if kind == "DENSE" else "DOT", "L2",
        frozenset({"TEXT_TO_TEXT"} if kind == "DENSE" else {"TEXT_TO_IMAGE", "IMAGE_TO_IMAGE"}))


class ServedBindingTests(unittest.IsolatedAsyncioTestCase):
    async def test_inference_uses_served_name_not_catalog_uuid(self):
        for kind in ("DENSE", "MULTIVECTOR"):
            space = spec(kind)
            receipt = dict(model_revision=space.revision, processor_revision=space.processor_revision,
                input_template=space.input_template, dimension=2, normalization="L2", similarity=space.similarity,
                directions=sorted(space.directions), vector=[1., 0.], vectors=[[1., 0.]], mask=[True])
            client = SimpleNamespace(represent=AsyncMock(return_value=receipt))
            gateway = KnowledgeModelGateway(dense_client=client, visual_client=client, auth_context=None,
                image_reader=None, models={space.space_id: {"model_ref":space.model_ref,
                    "served_model_name":"catalog-served-name"}}, max_elements=100)
            await gateway.represent(ModelInput("TEXT", text="查询"), space)
            payload = client.represent.await_args.args[0]
            self.assertEqual(payload["served_model_name"], "catalog-served-name")
            self.assertNotIn("model_ref", payload)

    async def test_missing_binding_is_rejected_before_network_request(self):
        space = spec()
        client = SimpleNamespace(represent=AsyncMock())
        gateway = KnowledgeModelGateway(dense_client=client, visual_client=client, auth_context=None,
            image_reader=None, models={space.space_id:{"model_ref":space.model_ref}}, max_elements=100)
        with self.assertRaises(AlgorithmError):
            await gateway.represent(ModelInput("TEXT", text="查询"), space)
        client.represent.assert_not_awaited()

    def test_configuration_requires_exact_explicit_bindings(self):
        space = spec()
        for names in ({}, {"other":"served"}, {space.model_ref:""}, {space.model_ref:" served "}):
            with self.subTest(names=names), self.assertRaises(AlgorithmError):
                KnowledgeFoundationConfig(representations=[encode(space)], served_model_names=names).profiles()
        config = KnowledgeFoundationConfig(representations=[encode(space)],
            served_model_names={space.model_ref:"served"})
        self.assertEqual(config.profiles()[1].representations, (space,))
        self.assertEqual(KnowledgeFoundationConfig().profiles()[1].representations, ())


if __name__ == "__main__":
    unittest.main()

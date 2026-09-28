from .standard import compile_standard
from platform_core.contracts.aiops import ImplementationProfile

def compile_patch(evidence, context):
    return compile_standard(ImplementationProfile.ORACLE_RU_PATCH, evidence, context)

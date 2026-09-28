from .standard import compile_standard
from platform_core.contracts.aiops import ImplementationProfile

def compile_clone(evidence, context):
    return compile_standard(ImplementationProfile.ORACLE_CLONE_REFRESH, evidence, context)

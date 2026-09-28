from .standard import compile_standard
from platform_core.contracts.aiops import ImplementationProfile

def compile_upgrade(evidence, context):
    return compile_standard(ImplementationProfile.ORACLE_DATABASE_UPGRADE, evidence, context)

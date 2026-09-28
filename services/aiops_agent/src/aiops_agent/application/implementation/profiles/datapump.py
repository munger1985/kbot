from .standard import compile_standard
from platform_core.contracts.aiops import ImplementationProfile

def compile_datapump(evidence, context):
    return compile_standard(ImplementationProfile.ORACLE_DATAPUMP_MIGRATION, evidence, context)

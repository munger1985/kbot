from .standard import compile_standard
from platform_core.contracts.aiops import ImplementationProfile

def compile_migration(evidence, context):
    return compile_standard(ImplementationProfile.ORACLE_DATABASE_MIGRATION, evidence, context)

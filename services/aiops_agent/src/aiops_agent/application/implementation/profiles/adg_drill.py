from .standard import compile_standard
from platform_core.contracts.aiops import ImplementationProfile

def compile_adg_drill(evidence, context):
    return compile_standard(ImplementationProfile.ORACLE_ADG_DRILL, evidence, context)

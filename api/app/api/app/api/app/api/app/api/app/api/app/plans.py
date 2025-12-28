from dataclasses import dataclass

@dataclass(frozen=True)
class PlanRules:
    seats_allowed: int
    sites_allowed: int
    exports_enabled: bool
    portfolio_enabled: bool
    monthly_analysis_included: int

PLAN_RULES = {
    "base": PlanRules(1, 1, False, False, 30),
    "pro": PlanRules(3, 10, True, True, 300),
    "enterprise": PlanRules(25, 9999, True, True, 999999),
}

def rules_for(plan: str) -> PlanRules:
    return PLAN_RULES.get(plan, PLAN_RULES["base"])

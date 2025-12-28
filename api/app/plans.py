from dataclasses import dataclass

@dataclass(frozen=True)
class PlanRules:
    runs_per_month: int
    exports_enabled: bool
    portfolio_enabled: bool

PLAN_RULES = {
    "base": PlanRules(runs_per_month=30, exports_enabled=False, portfolio_enabled=False),
    "pro": PlanRules(runs_per_month=300, exports_enabled=True, portfolio_enabled=True),
    "enterprise": PlanRules(runs_per_month=999999, exports_enabled=True, portfolio_enabled=True),
}

def rules_for(plan: str) -> PlanRules:
    return PLAN_RULES.get(plan, PLAN_RULES["base"])

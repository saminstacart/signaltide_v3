"""
SignalTide V5 Configuration Module.

Based on Phase 6 Deep Research findings and Data Availability Audit.
Key changes from V4:
- 40 positions (up from 25) to reduce idiosyncratic risk
- Top 20% entry (up from Top 10%) for robustness
- Hierarchical signal logic: Quality filter -> Insider alpha -> Momentum timing
- Quarterly rebalancing option (reduced turnover)
- Removed 2-month minimum holding constraint

References:
- Phase 6 Final Verdict (2025-12-01)
- Sharadar Data Availability Audit (S&P 500 only)
"""

from typing import Dict, Any


# V5 Configuration - Phase 6 Research-Based Parameters
V5_CONFIG: Dict[str, Any] = {
    # Universe - confirmed by data audit (S&P 500 only available)
    'universe': {
        'source': 'sp500_actual',
        'table': 'dim_universe_membership',
        'pit_compliant': True,
        'rationale': 'Only reliable point-in-time data available (S&P 400/600 not in Sharadar)',
    },

    # Portfolio construction - Phase 6 recommendations
    'portfolio': {
        'target_positions': 40,           # Up from 25 (reduce idiosyncratic risk)
        'max_position_weight': 0.04,      # 4% max (adjusted for 40 positions)
        'entry_percentile': 20,           # Up from 10% (plateau vs peak robustness)
        'exit_percentile': 50,            # Keep wide buffer (unchanged)
        'min_holding_months': 0,          # Removed (was 2 months - over-constraining)
    },

    # Signal configuration - Hierarchical logic (new in V5)
    'signals': {
        'logic': 'hierarchical',  # NOT 'equal_weight' as in V4

        # Stage 1: Quality Filter (safety gate)
        'quality': {
            'role': 'filter',
            'pass_percentile': 40,       # Top 40% pass quality gate
            'components': ['cbop', 'bab', 'asset_growth'],
            'rationale': 'Cash profitability + low beta + conservative investment',
        },

        # Stage 2: Insider Signal (alpha source)
        'insider': {
            'role': 'alpha',
            'weight': 0.6,               # 60% weight in final score
            'date_field': 'filingdate',  # CRITICAL - verified no look-ahead bias
            'lookback_days': 180,
            'filters': ['opportunistic_only', 'cluster_bonus'],
            'rationale': 'Cohen-Malloy-Pomorski methodology - genuine edge for retail',
        },

        # Stage 3: Momentum Signal (timing)
        'momentum': {
            'role': 'timing',
            'weight': 0.4,               # 40% weight in final score
            'type': 'residual',          # FF3-adjusted momentum
            'formation_months': 12,
            'skip_recent_month': True,   # Jegadeesh-Titman methodology
            'rationale': 'Residual momentum less crowded than raw momentum',
        },
    },

    # Rebalancing - test both
    'rebalancing': {
        'frequency': 'monthly',          # Primary (match V4 baseline)
        'alternative': 'quarterly',      # Test for turnover reduction
        'months': [3, 6, 9, 12],         # Quarter-end months
    },

    # Risk constraints
    'constraints': {
        'max_sector_weight': 0.35,       # Hard sector cap
        'beta_constraint': 'soft',       # Inverse-vol weighting (no hard cap)
        'signal_smoothing': 3,           # 3-period exponential smoothing
    },

    # Validation requirements (must meet to approve V5)
    'validation': {
        'walk_forward_folds': 5,
        'min_oos_sharpe': 0.3,
        'max_train_test_corr_magnitude': 0.6,  # Fail if |corr| > 0.6
    },
}


# V4 Baseline Configuration (for comparison)
V4_CONFIG: Dict[str, Any] = {
    'universe': {
        'source': 'sp500_actual',
        'table': 'dim_universe_membership',
        'pit_compliant': True,
    },
    'portfolio': {
        'target_positions': 25,
        'max_position_weight': 0.05,
        'entry_percentile': 10,
        'exit_percentile': 50,
        'min_holding_months': 2,
    },
    'signals': {
        'logic': 'equal_weight',
        'quality': {'weight': 0.33},
        'insider': {'weight': 0.33},
        'momentum': {'weight': 0.34},
    },
    'rebalancing': {
        'frequency': 'monthly',
    },
    'constraints': {
        'max_sector_weight': 0.35,
        'signal_smoothing': 3,
    },
}


# Comparison matrix configurations for A/B testing
COMPARISON_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Baseline: V4 parameters
    'V4-Baseline': {
        **V5_CONFIG,
        'portfolio': V4_CONFIG['portfolio'].copy(),
        'signals': V4_CONFIG['signals'].copy(),
    },

    # Test 1: Only positions increased
    'V5-Positions40': {
        **V5_CONFIG,
        'portfolio': {
            **V4_CONFIG['portfolio'],
            'target_positions': 40,
            'max_position_weight': 0.04,
        },
        'signals': V4_CONFIG['signals'].copy(),
    },

    # Test 2: Only entry threshold relaxed
    'V5-Entry20': {
        **V5_CONFIG,
        'portfolio': {
            **V4_CONFIG['portfolio'],
            'entry_percentile': 20,
        },
        'signals': V4_CONFIG['signals'].copy(),
    },

    # Test 3: Only quarterly rebalancing
    'V5-Quarterly': {
        **V5_CONFIG,
        'portfolio': V4_CONFIG['portfolio'].copy(),
        'signals': V4_CONFIG['signals'].copy(),
        'rebalancing': {'frequency': 'quarterly'},
    },

    # Test 4a: Hierarchical scoring with HARD quality gate
    'V5-Hierarchical-Hard': {
        **V5_CONFIG,
        'portfolio': V4_CONFIG['portfolio'].copy(),
        'rebalancing': {'frequency': 'monthly'},
        'scorer': {
            'mode': 'hard',
            'quality_threshold_percentile': 40,
            'insider_weight': 0.6,
            'momentum_weight': 0.4,
        },
    },

    # Test 4b: Hierarchical scoring with SOFT quality gate
    # This preserves strong insider signals even in lower-quality stocks
    'V5-Hierarchical-Soft': {
        **V5_CONFIG,
        'portfolio': V4_CONFIG['portfolio'].copy(),
        'rebalancing': {'frequency': 'monthly'},
        'scorer': {
            'mode': 'soft',
            'quality_threshold_percentile': 40,
            'soft_gate_multiplier': 0.5,  # 50% penalty for failing quality
            'insider_weight': 0.6,
            'momentum_weight': 0.4,
        },
    },

    # Test 5: Full V5 with HARD gate (all changes combined)
    'V5-Full-Hard': {
        **V5_CONFIG,
        'scorer': {
            'mode': 'hard',
            'quality_threshold_percentile': 40,
            'insider_weight': 0.6,
            'momentum_weight': 0.4,
        },
    },

    # Test 6: Full V5 with SOFT gate (all changes combined)
    'V5-Full-Soft': {
        **V5_CONFIG,
        'scorer': {
            'mode': 'soft',
            'quality_threshold_percentile': 40,
            'soft_gate_multiplier': 0.5,
            'insider_weight': 0.6,
            'momentum_weight': 0.4,
        },
    },

    # Legacy alias: V5-Full points to Hard gate (Phase 6 recommendation)
    'V5-Full': V5_CONFIG,
}


def get_config(name: str = 'V5-Full') -> Dict[str, Any]:
    """
    Get configuration by name.

    Args:
        name: Configuration name (V4-Baseline, V5-Full, etc.)

    Returns:
        Configuration dictionary
    """
    if name in COMPARISON_CONFIGS:
        return COMPARISON_CONFIGS[name]
    elif name == 'V5':
        return V5_CONFIG
    elif name == 'V4':
        return V4_CONFIG
    else:
        raise ValueError(f"Unknown config: {name}. Available: {list(COMPARISON_CONFIGS.keys())}")


def summarize_config_diff(config1_name: str, config2_name: str) -> str:
    """
    Summarize differences between two configurations.

    Args:
        config1_name: First config name
        config2_name: Second config name

    Returns:
        Human-readable diff summary
    """
    c1 = get_config(config1_name)
    c2 = get_config(config2_name)

    diffs = []

    # Portfolio diffs
    for key in ['target_positions', 'entry_percentile', 'exit_percentile', 'min_holding_months']:
        v1 = c1.get('portfolio', {}).get(key)
        v2 = c2.get('portfolio', {}).get(key)
        if v1 != v2:
            diffs.append(f"  {key}: {v1} -> {v2}")

    # Signal logic diff
    logic1 = c1.get('signals', {}).get('logic')
    logic2 = c2.get('signals', {}).get('logic')
    if logic1 != logic2:
        diffs.append(f"  signal_logic: {logic1} -> {logic2}")

    # Rebalancing diff
    freq1 = c1.get('rebalancing', {}).get('frequency')
    freq2 = c2.get('rebalancing', {}).get('frequency')
    if freq1 != freq2:
        diffs.append(f"  rebalancing: {freq1} -> {freq2}")

    if diffs:
        return f"{config1_name} vs {config2_name}:\n" + "\n".join(diffs)
    else:
        return f"{config1_name} == {config2_name} (no differences)"


if __name__ == '__main__':
    print("SignalTide V5 Configuration Module")
    print("=" * 50)

    print("\nV5 Key Changes from V4:")
    print(summarize_config_diff('V4-Baseline', 'V5-Full'))

    print("\nAvailable Configurations:")
    for name in COMPARISON_CONFIGS:
        print(f"  - {name}")

    print("\nV5 Full Configuration:")
    import json
    print(json.dumps(V5_CONFIG, indent=2, default=str))

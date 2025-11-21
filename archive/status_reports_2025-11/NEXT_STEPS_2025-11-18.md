# Next Steps - SignalTide v3

**Last Updated:** 2025-11-18

This document contains the prioritized task list for SignalTide v3 development.

---

## Immediate Priorities (Current Sprint)

### 1. Complete Initial Infrastructure Setup
- [ ] Finish creating all documentation files
- [ ] Set up Python package structure
- [ ] Initialize git repository
- [ ] Create comprehensive .gitignore

### 2. Implement Core Base Classes
- [ ] BaseSignal abstract class with required interface
- [ ] Portfolio class for position management
- [ ] DataManager for all data access operations
- [ ] Ensure no lookahead bias in base implementations

### 3. Set Up Validation Framework
- [ ] PurgedKFold cross-validation implementation
- [ ] MonteCarloValidator for permutation tests
- [ ] StatisticalTests class for significance testing
- [ ] DeflatedSharpe calculator
- [ ] Write tests for validation framework

### 4. Create Optimization Infrastructure
- [ ] OptunaManager class with parallel execution
- [ ] ParameterSpace class to parse HYPERPARAMETERS.md
- [ ] Study persistence to SQLite
- [ ] Automatic overfitting detection
- [ ] Write tests for optimization framework

---

## Next Phase: Signal Migration

### 5. Migrate First Signal (Proof of Concept)
- [ ] Choose simplest existing signal to migrate
- [ ] Implement as subclass of BaseSignal
- [ ] Document all assumptions
- [ ] Define parameter ranges in HYPERPARAMETERS.md
- [ ] Run through full validation pipeline
- [ ] Document results in signal-specific README

### 6. Establish Signal Migration Pattern
- [ ] Create signal migration template
- [ ] Document migration process
- [ ] Create migration checklist
- [ ] Test with second signal

---

## Future Phases (Not Yet Started)

### 7. Backtest Engine
- [ ] Implement vectorized backtesting
- [ ] Transaction cost modeling
- [ ] Slippage modeling
- [ ] Realistic order execution simulation
- [ ] Performance metrics calculation

### 8. Data Infrastructure
- [ ] Set up SQLite database schema
- [ ] Implement data ingestion pipeline
- [ ] Add data quality checks
- [ ] Create data versioning system
- [ ] Implement caching for performance

### 9. Risk Management
- [ ] Position sizing algorithms
- [ ] Stop-loss implementations
- [ ] Drawdown monitoring
- [ ] Regime detection framework
- [ ] Risk-adjusted position scaling

### 10. Regime Detection (Optional Enhancement)
- [ ] Research regime detection methods
- [ ] Implement regime classification
- [ ] Add regime-aware signal weighting
- [ ] Validate regime detection doesn't overfit

### 11. Production Deployment
- [ ] Live trading interface design
- [ ] Paper trading mode
- [ ] Monitoring and alerting
- [ ] Performance tracking dashboard
- [ ] Incident response procedures

---

## Deferred / Future Considerations

- Multi-asset expansion (beyond crypto)
- Real-time data feeds
- Cloud deployment
- Web interface for monitoring
- Advanced ML signal integration
- Alternative data sources

---

## Success Criteria

Before moving to the next phase, ensure:

1. All tests pass
2. Documentation is complete and accurate
3. Code follows single-responsibility principle
4. No lookahead bias detected
5. Validation framework confirms no overfitting
6. Code review completed
7. Performance is acceptable (even if not optimal)

---

## Anti-Patterns to Avoid

- Manual parameter tuning (use Optuna)
- Filtering parameter ranges prematurely
- Adding complexity before validating simplicity
- Skipping validation steps
- Creating files "just in case"
- Optimizing for speed before correctness
- Adding features before validating existing ones

---

## Questions to Resolve

None currently - will be updated as development progresses.

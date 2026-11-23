# API Reference

Every exported name, and nothing else.

```@docs
DampedLinearSolve
```

## Systems and their construction

```@docs
AbstractLinearSystem
makehessiansystem
makejacobiansystem
structurekey
checkhessian
checkjacobian
check_jacobian_linsolver
jacobian_ncols
jacobian_nrows
checklinsolver
validatelinsolver
```

## Filling

```@docs
getA
getb
getx
zero!
getJ
getr
gradient!
markstale!
```

## Damping and solving

```@docs
dampen!
solve!
SolveStatus
Success
NotPositiveDefinite
Indefinite
```

## Queries on the undamped system

```@docs
max_diag
getdiag!
quadform
quadratic_model_change
dampingfloor
isfinitesystem
systemeltype
```

## The spare system

```@docs
copy_structure
sharex
```

## Conformance

```@docs
check_linear_system
conformance_matrix
conformance_fill!
conformance_fillb!
conformance_traits
```

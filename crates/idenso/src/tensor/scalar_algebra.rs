//! Reversible scalar-algebra inputs on established tensor interfaces.

use std::{
    cell::Cell,
    collections::{BTreeMap, HashSet},
    sync::atomic::{AtomicU64, Ordering},
};

use spenso::{
    network::{library::function_lib::INBUILTS, tags::SPENSO_TAG},
    shadowing::TensorCollectFilter,
    structure::{TensorStructure, partial::PartialStructure},
};
use symbolica::atom::{
    AliasedAtom, Atom, AtomCore, AtomView, NamespacedSymbol, Symbol, SymbolAttribute, SymbolBuilder,
};

use super::{
    ScalarPayloadPosition, SymbolicTensor,
    inference::{InterfaceInference, TensorInferenceError},
};

type Result<T> = std::result::Result<T, TensorInferenceError>;

fn selected_closed_payload(
    position: ScalarPayloadPosition<'_>,
    select: &mut impl FnMut(ScalarPayloadPosition<'_>) -> bool,
    failure: &mut Option<TensorInferenceError>,
) -> bool {
    let value = match position {
        ScalarPayloadPosition::Root { .. } => return false,
        _ if failure.is_some() || !select(position) => return false,
        ScalarPayloadPosition::FunctionArgument { function, argument } => {
            function.iter().nth(argument).unwrap()
        }
        ScalarPayloadPosition::PowerBase { power } => power.get_base_exp().0,
        ScalarPayloadPosition::PowerExponent { power } => power.get_base_exp().1,
        ScalarPayloadPosition::ClosedBracket { function } => AtomView::Fun(function),
    };
    // Only role-selected mathematical positions are observed. Valid open
    // tensor powers and conjugated component fields remain native; unselected
    // metadata is never entered.
    match InterfaceInference::replacement_interface(value) {
        Ok(interface) => interface.canonical().is_scalar(),
        Err(error) => {
            *failure = Some(error);
            false
        }
    }
}

struct ScalarAlgebraInput {
    variable: Option<Symbol>,
    reserved: HashSet<Symbol>,
    aliases: BTreeMap<Atom, Atom>,
}

impl ScalarAlgebraInput {
    fn new(source: &SymbolicTensor<PartialStructure>, variable: Option<Symbol>) -> Self {
        Self {
            variable,
            reserved: source
                .expression()
                .get_all_symbols(true)
                .into_iter()
                .collect(),
            aliases: BTreeMap::new(),
        }
    }

    fn contains_variable(&self, value: &Atom) -> bool {
        self.variable
            .is_some_and(|variable| value.get_all_symbols(true).contains(&variable))
    }

    fn contains_word(value: &Atom) -> bool {
        value
            .get_all_symbols(true)
            .iter()
            .any(|symbol| symbol.has_tag(&SPENSO_TAG.tensor_word))
    }

    fn reject_packed_variable(&self, value: &Atom) -> Result<()> {
        let Some(variable) = self.variable else {
            return Ok(());
        };
        let hidden = Cell::new(false);
        value.visitor(&mut |node| {
            if matches!(node, AtomView::Num(_)) && node.get_all_symbols(true).contains(&variable) {
                hidden.set(true);
            }
            !hidden.get()
        });
        if hidden.get() {
            return Err(TensorInferenceError::invalid(
                "scalar algebra with its expansion variable hidden in a numeric coefficient encoding is unsupported; expose that coefficient through the native coefficient API before preparation",
            ));
        }
        Ok(())
    }

    fn alias(&mut self, original: Atom) -> Result<Atom> {
        if let Some(alias) = self.aliases.get(&original) {
            return Ok(alias.clone());
        }
        let alias = loop {
            // Reuse the native component parser's globally fresh symbol
            // pattern. build_group holds the Symbolica state lock and refuses
            // existing symbols, including identities with assumptions/hooks.
            static NEXT_ALIAS: AtomicU64 = AtomicU64::new(0);
            let serial = loop {
                let current = NEXT_ALIAS.load(Ordering::Relaxed);
                let next = current.checked_add(1).ok_or_else(|| {
                    TensorInferenceError::invalid("scalar-algebra alias index overflow")
                })?;
                if NEXT_ALIAS
                    .compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
                {
                    break current;
                }
            };
            let name =
                NamespacedSymbol::parse(&format!("idenso::scalar_algebra_coefficient_{}", serial));
            let Ok(symbols) = SymbolBuilder::build_group(vec![
                SymbolBuilder::new(name).with_attributes(vec![SymbolAttribute::Scalar]),
            ]) else {
                continue;
            };
            let symbol = symbols[0];
            if self.reserved.insert(symbol) {
                break Atom::var(symbol);
            }
        };
        self.aliases.insert(original, alias.clone());
        Ok(alias)
    }

    fn prepare(
        &mut self,
        source: &SymbolicTensor<PartialStructure>,
        nonlinear: bool,
    ) -> Result<SymbolicTensor<PartialStructure>> {
        if source.expression().is_zero() {
            return Ok(source.clone());
        }
        if nonlinear && Self::contains_word(source.expression()) {
            return Err(TensorInferenceError::invalid(
                "nonlinear scalar algebra on native tensor words is unsupported; retain the native word algebra",
            ));
        }
        let mut terms = Vec::new();
        for (selected, coefficient) in source.coefficient_list::<0>(TensorCollectFilter::Tensors)? {
            if self.contains_variable(selected.expression()) {
                return Err(TensorInferenceError::invalid(
                    "scalar-algebra preparation retains its variable inside a tensor payload; select the mathematical closed payload explicitly",
                ));
            }
            let selected = if selected.is_scalar()
                && !selected.expression().is_one()
                && !Self::contains_word(selected.expression())
            {
                let protected = SymbolicTensor::<PartialStructure>::scalar_power_expression(
                    selected.expression().clone(),
                );
                if self.contains_variable(&protected) {
                    return Err(TensorInferenceError::invalid(
                        "a scalar-algebra alias definition contains its expansion variable",
                    ));
                }
                self.alias(protected)?
            } else {
                selected.expression().clone()
            };
            terms.push(selected * coefficient.expression());
        }
        source.with_checked_expression(Atom::add_many(terms))
    }
}

impl SymbolicTensor<PartialStructure> {
    /// Prepare reversible input for scalar algebra, retaining native tensor scopes.
    ///
    /// Selected mathematical payloads use [`Self::map_closed_scalar_payloads`].
    /// Existing native collection separates their tensor sectors from scalar
    /// coefficients; only variable-independent closed ordinary sectors become
    /// aliases. Alias definitions use the existing native scalar-power wrapper, so
    /// scalar powers restore closed contractions rather than powers of open
    /// tensor factors. Expansion variables remain visible in the root.
    ///
    /// The selector must identify mathematical function arguments, bases and
    /// exponents; arbitrary scalar-tagged metadata is not mathematical input.
    /// Open native tensor bases/brackets remain native. Root is prepared once
    /// using its original typed interface, including typed zeros and open ports.
    /// Explicitly selected unary conjugation arguments are entered only when
    /// independently closed; open conjugated tensors retain their native roles.
    /// The general payload mapper's default conjugation opacity is unchanged.
    /// Native words remain under their existing algebra; nonlinear word payloads
    /// are explicitly unsupported.
    ///
    /// Numeric coefficient encodings hiding the expansion variable are refused.
    /// Run scalar algebra on the returned [`AliasedAtom::get_root`], then use
    /// [`Self::with_scalar_algebra_result`] to restore the unmodified registry
    /// before checked tensor publication. Temporary aliases must not be
    /// published or persisted.
    pub fn scalar_algebra_input(
        &self,
        variable: Symbol,
        select_payload: impl FnMut(ScalarPayloadPosition<'_>) -> bool,
    ) -> Result<AliasedAtom> {
        self.prepare_scalar_algebra_input(Some(variable), select_payload)
    }

    /// Prepare the same scoped algebra without requiring a distinguished variable.
    /// An absent variable changes only variable-presence checks; aliases still
    /// reserve every source symbol and use the existing checked restoration.
    pub(super) fn prepare_scalar_algebra_input(
        &self,
        variable: Option<Symbol>,
        mut select_payload: impl FnMut(ScalarPayloadPosition<'_>) -> bool,
    ) -> Result<AliasedAtom> {
        self.ensure_validated()?;
        let mut input = ScalarAlgebraInput::new(self, variable);
        input.reject_packed_variable(self.expression())?;
        let mut failure = None;
        let prepared = self.map_scalar_algebra_payloads(
            |position| selected_closed_payload(position, &mut select_payload, &mut failure),
            |position, value| {
                input.prepare(
                    value,
                    matches!(
                        position,
                        ScalarPayloadPosition::FunctionArgument { .. }
                            | ScalarPayloadPosition::PowerBase { .. }
                            | ScalarPayloadPosition::PowerExponent { .. }
                    ),
                )
            },
        )?;
        if let Some(error) = failure {
            return Err(error);
        }
        let prepared = input.prepare(&prepared, false)?;
        let mut result = AliasedAtom::from(prepared.expression().clone());
        for (original, alias) in input.aliases {
            // Match the complete conjugated alias before ordinary substitution
            // can normalize a restored tensor inside Symbolica's scalar CONJ.
            result.register_alias(alias.conj(), INBUILTS.conj(&original));
            result.register_alias(alias, original);
        }
        Ok(result)
    }

    /// Restore a scalar-algebra result and validate this tensor's original interface.
    ///
    /// `input` must be the unmodified registry returned by
    /// [`Self::scalar_algebra_input`]. Restoration uses Symbolica's existing
    /// alias substitution. Complete symbol discovery then refuses any alias
    /// still hidden in a numeric coefficient encoding, rather than publishing
    /// an unresolved result. This does not decode or rewrite coefficients.
    /// Supply the same mathematical-payload selection policy as preparation.
    /// The native product normalizer then removes redundant brackets only at
    /// those independent payload roots, retaining protected power bases and
    /// unselected metadata exactly.
    pub fn with_scalar_algebra_result(
        &self,
        input: &AliasedAtom,
        expression: Atom,
        mut select_payload: impl FnMut(ScalarPayloadPosition<'_>) -> bool,
    ) -> Result<Self> {
        let restored = input.clone().map_root(|_| expression).into_inner();
        let symbols = restored.get_all_symbols(true);
        // Each fresh identity retains its original Var key. Composite alias
        // keys (such as CONJ(alias)) also contain universal structural heads,
        // which are not temporary identities and may occur in legitimate output.
        if input.get_aliases().keys().any(|alias| {
            matches!(alias.as_view(), AtomView::Var(variable)
                if symbols.contains(&variable.get_symbol()))
        }) {
            return Err(TensorInferenceError::invalid(
                "scalar-algebra restoration left a temporary alias in a numeric coefficient encoding; expose that coefficient through the native coefficient API before restoration",
            ));
        }
        let restored = self.with_checked_expression(restored)?;
        let mut failure = None;
        let normalized = restored.map_scalar_algebra_payloads(
            |position| selected_closed_payload(position, &mut select_payload, &mut failure),
            |_, value| {
                value.with_checked_expression(
                    Self::normalize_products(value.expression().as_view()).into_owned(),
                )
            },
        )?;
        if let Some(error) = failure {
            return Err(error);
        }
        self.with_checked_expression(normalized.into_expression())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, atomic::AtomicUsize};

    use super::*;
    use crate::{representations::ColorFundamental, test_support::test_initialize};
    use spenso::{
        algebra::ScalarMul,
        structure::{
            partial::PartialStructureExt,
            representation::{Minkowski, RepName},
        },
    };
    use symbolica::{
        coefficient::Coefficient,
        domains::{atom::AtomField, integer::Z, rational::Q},
        function,
        poly::series::{Series, SeriesDepth},
        symbol,
    };

    type Tensor = SymbolicTensor<PartialStructure>;

    fn mathematical_position(position: ScalarPayloadPosition<'_>) -> bool {
        match position {
            ScalarPayloadPosition::Root { .. } => false,
            ScalarPayloadPosition::FunctionArgument { function, .. } => {
                [Symbol::SIN, Symbol::COS, Symbol::EXP, Symbol::LOG]
                    .contains(&function.get_symbol())
            }
            _ => true,
        }
    }

    fn pair(name: &str, index: usize) -> Tensor {
        test_initialize();
        let slot = ColorFundamental {}.to_symbolic([Atom::num(3), Atom::num(index)]);
        Tensor::infer(
            function!(
                SPENSO_TAG.tensor_symbol(&format!("scalar_algebra::{name}_bar")),
                spenso::dind!(slot.clone())
            ) * function!(
                SPENSO_TAG.tensor_symbol(&format!("scalar_algebra::{name}")),
                slot
            ),
        )
        .unwrap()
    }

    fn constant(value: i64) -> Tensor {
        Tensor::infer(Atom::num(value)).unwrap()
    }

    fn series(source: &Tensor, variable: Symbol, order: i64) -> (AliasedAtom, Series<AtomField>) {
        let input = source
            .scalar_algebra_input(variable, mathematical_position)
            .unwrap();
        assert!(
            input
                .get_aliases()
                .values()
                .all(|body| !body.get_all_symbols(true).contains(&variable))
        );
        let series = input
            .get_root()
            .series(variable, 0, SeriesDepth::absolute(order))
            .unwrap();
        (input, series)
    }

    fn coefficient(
        source: &Tensor,
        input: &AliasedAtom,
        series: &Series<AtomField>,
        grade: i64,
    ) -> Tensor {
        let coefficient = series.coefficient(grade.into()).unwrap();
        let result = source
            .with_scalar_algebra_result(input, coefficient, mathematical_position)
            .unwrap();
        assert!(!result.expression().contains_symbol(Symbol::DERIVATIVE));
        let symbols = result.expression().get_all_symbols(true);
        assert!(input.get_aliases().keys().all(|alias| {
            !matches!(alias.as_view(), AtomView::Var(variable)
                if symbols.contains(&variable.get_symbol()))
        }));
        result
    }

    fn assert_equal(actual: &Tensor, expected: &Tensor) {
        assert_eq!(actual.structure(), expected.structure());
        assert_eq!(
            actual.canonize().unwrap().expression(),
            expected.canonize().unwrap().expression()
        );
    }

    #[test]
    fn closed_nonselfdual_powers_and_inverse_coefficients_restore_native_scopes() {
        let p = pair("power_p", 1);
        let variable = symbol!("scalar_algebra::power_t"; Scalar, Real, Positive);
        let t = Atom::var(variable);
        let marked = p
            .with_checked_expression(spenso::bracket!(t.pow(2) * p.expression()).pow(2))
            .unwrap();
        let (input, expansion) = series(&marked, variable, 4);
        assert!(coefficient(&marked, &input, &expansion, 0).is_zero());
        assert_equal(
            &coefficient(&marked, &input, &expansion, 4),
            &p.pow(&constant(2)).unwrap(),
        );

        let mass = Atom::var(symbol!("scalar_algebra::mass"; Real));
        let inverse = p
            .with_checked_expression((mass.pow(2) + t.pow(2) * p.expression()).pow(-1))
            .unwrap();
        let (input, expansion) = series(&inverse, variable, 4);
        for (grade, expected) in [
            (0, Tensor::infer(mass.pow(-2)).unwrap()),
            (2, p.scalar_mul(&(-mass.pow(-4))).unwrap()),
            (
                4,
                p.pow(&constant(2))
                    .unwrap()
                    .scalar_mul(&mass.pow(-6))
                    .unwrap(),
            ),
        ] {
            assert_equal(&coefficient(&inverse, &input, &expansion, grade), &expected);
        }
    }

    #[test]
    fn distinct_unequal_weight_closed_sums_and_nested_analytic_payloads_remain_visible() {
        let p = pair("mixed_p", 2);
        let q = pair("mixed_q", 3);
        let variable = symbol!("scalar_algebra::mixed_t"; Scalar, Real, Positive);
        let t = Atom::var(variable);
        let mixed = p
            .with_checked_expression(
                spenso::bracket!(t.pow(2) * p.expression() + t.pow(3) * q.expression()).pow(2),
            )
            .unwrap();
        let (input, expansion) = series(&mixed, variable, 6);
        assert_equal(
            &coefficient(&mixed, &input, &expansion, 4),
            &p.pow(&constant(2)).unwrap(),
        );
        assert_equal(
            &coefficient(&mixed, &input, &expansion, 5),
            &p.multiply(&q).unwrap().scalar_mul(&Atom::num(2)).unwrap(),
        );
        assert_equal(
            &coefficient(&mixed, &input, &expansion, 6),
            &q.pow(&constant(2)).unwrap(),
        );

        let analytic = p
            .with_checked_expression((t.pow(2) * p.expression()).cos())
            .unwrap();
        let (input, expansion) = series(&analytic, variable, 4);
        assert_equal(&coefficient(&analytic, &input, &expansion, 0), &constant(1));
        assert_equal(
            &coefficient(&analytic, &input, &expansion, 4),
            &p.pow(&constant(2))
                .unwrap()
                .scalar_mul(&Atom::num((-1, 2)))
                .unwrap(),
        );
        let unweighted = p.with_checked_expression(p.expression().cos()).unwrap();
        let (input, expansion) = series(&unweighted, variable, 0);
        assert_equal(
            &coefficient(&unweighted, &input, &expansion, 0),
            &unweighted,
        );
        let nested = p
            .with_checked_expression((t.pow(2) * p.expression()).sin().sin())
            .unwrap();
        let (input, expansion) = series(&nested, variable, 6);
        assert_equal(&coefficient(&nested, &input, &expansion, 2), &p);
        assert_equal(
            &coefficient(&nested, &input, &expansion, 6),
            &p.pow(&constant(3))
                .unwrap()
                .scalar_mul(&Atom::num((-1, 3)))
                .unwrap(),
        );
    }

    #[test]
    fn large_closed_power_is_protected_before_a_low_order_series() {
        let p = pair("large_p", 4);
        let variable = symbol!("scalar_algebra::large_t"; Scalar, Real, Positive);
        let marked = p
            .with_checked_expression(spenso::bracket!(
                Atom::var(variable).pow(2) * p.expression()
            ))
            .unwrap();
        let marked = marked.pow(&constant(10_000)).unwrap();
        let (input, expansion) = series(&marked, variable, 6);
        assert!(input.get_aliases().len() <= 2);
        assert!(expansion.to_atom().is_zero());
        assert!(coefficient(&marked, &input, &expansion, 6).is_zero());
    }

    #[test]
    fn open_spectators_and_ranked_zeros_keep_their_native_interfaces() {
        let p = pair("spectator_p", 5);
        let slot = ColorFundamental {}.to_symbolic([Atom::num(3), Atom::num(50)]);
        let open = Tensor::infer(function!(
            SPENSO_TAG.tensor_symbol("scalar_algebra::open"),
            slot
        ))
        .unwrap();
        let expected = p.pow(&constant(2)).unwrap().multiply(&open).unwrap();
        let variable = symbol!("scalar_algebra::spectator_t"; Scalar, Real, Positive);
        let marked = expected.scalar_mul(&Atom::var(variable).pow(4)).unwrap();
        let (input, expansion) = series(&marked, variable, 4);
        assert_equal(&coefficient(&marked, &input, &expansion, 4), &expected);
        let zero = open.scalar_mul(&Atom::Zero).unwrap();
        let (input, expansion) = series(&zero, variable, 4);
        let restored = coefficient(&zero, &input, &expansion, 0);
        assert!(restored.is_zero());
        assert_eq!(restored.structure(), zero.structure());
        assert_eq!(restored.rank(), 1);

        // A representation-only port is established native shape, even though
        // its syntax has no explicit abstract index. It must never be aliased
        // as a closed coefficient during preparation.
        let anonymous = Tensor::infer(function!(
            SPENSO_TAG.tensor_symbol("scalar_algebra::anonymous_vector"),
            spenso::mink!(4)
        ))
        .unwrap();
        assert_eq!(anonymous.rank(), 1);
        let marked = anonymous.scalar_mul(&Atom::var(variable).pow(2)).unwrap();
        let (input, expansion) = series(&marked, variable, 2);
        assert!(input.get_aliases().is_empty());
        let restored = coefficient(&marked, &input, &expansion, 2);
        assert_eq!(restored.structure(), anonymous.structure());
        assert_eq!(restored.expression(), anonymous.expression());
    }

    fn pack(value: &Atom) -> Atom {
        Atom::num(Coefficient::RationalPolynomial(
            value.to_rational_polynomial::<_, _, u16>(&Q, &Z, None),
        ))
    }

    #[test]
    fn packed_variables_and_unrestored_packed_aliases_are_explicitly_refused() {
        let p = pair("packed_p", 6);
        let variable = symbol!("scalar_algebra::packed_t"; Scalar, Real, Positive);
        for expression in [pack(&Atom::var(variable)), pack(&Atom::var(variable)).sin()] {
            let source = Tensor::infer(expression).unwrap();
            let error = source
                .scalar_algebra_input(variable, |_| true)
                .unwrap_err()
                .to_string();
            assert!(error.contains("hidden in a numeric coefficient"));
        }
        let marked = p.scalar_mul(&Atom::var(variable).pow(2)).unwrap();
        let (input, _) = series(&marked, variable, 2);
        let alias = input.get_aliases().keys().next().unwrap();
        let error = marked
            .with_scalar_algebra_result(&input, pack(alias), mathematical_position)
            .unwrap_err()
            .to_string();
        assert!(error.contains("temporary alias"));
        let mass = Atom::var(symbol!("scalar_algebra::packed_mass"));
        let packed_coefficient = marked.scalar_mul(&pack(&mass)).unwrap();
        let (input, expansion) = series(&packed_coefficient, variable, 2);
        assert_equal(
            &coefficient(&packed_coefficient, &input, &expansion, 2),
            &p.scalar_mul(&pack(&mass)).unwrap(),
        );
        assert!(!expansion.to_atom().is_zero());
    }

    #[test]
    fn alias_like_metadata_and_independent_preparations_are_not_captured() {
        let p = pair("metadata_p", 7);
        let variable = symbol!("scalar_algebra::metadata_t"; Scalar, Real, Positive);
        let literal = Atom::var(symbol!("idenso::scalar_algebra_coefficient_0"));
        let dimension = Atom::var(symbol!("scalar_algebra::symbolic_dimension"));
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = calls.clone();
        let head = symbol!("scalar_algebra::opaque_metadata"; Scalar; norm=move |_, _| {
            observed.fetch_add(1, Ordering::SeqCst);
        });
        let metadata = function!(
            head,
            literal.clone(),
            Minkowski {}.to_symbolic([dimension, literal.clone()]),
            p.expression()
        );
        let marked = p
            .scalar_mul(&(Atom::var(variable).pow(2) * &metadata))
            .unwrap();
        let expected = p.scalar_mul(&metadata).unwrap();
        let before = calls.load(Ordering::SeqCst);
        let make = || {
            marked
                .scalar_algebra_input(variable, mathematical_position)
                .unwrap()
        };
        let first = make();
        let second = make();
        assert!(first.get_aliases().keys().all(|alias| {
            !alias
                .get_all_symbols(true)
                .iter()
                .any(|symbol| literal.get_all_symbols(true).contains(symbol))
        }));
        assert!(
            first
                .get_aliases()
                .keys()
                .all(|alias| !second.get_aliases().contains_key(alias))
        );
        for input in [first, second] {
            let series = input
                .get_root()
                .series(variable, 0, SeriesDepth::absolute(2))
                .unwrap();
            let restored = coefficient(&marked, &input, &series, 2);
            assert!(restored.expression().contains(&metadata));
            // This is an exact opacity/roundtrip contract, independent of
            // canonicalizing inside mathematical scalar payloads.
            assert_eq!(restored.structure(), expected.structure());
            assert_eq!(restored.expression(), expected.expression());
        }
        assert_eq!(calls.load(Ordering::SeqCst), before);
    }

    #[test]
    fn globally_preoccupied_alias_assumptions_and_hooks_are_never_adopted() {
        const CHILD: &str = "IDENSO_SCALAR_ALGEBRA_ALIAS_OCCUPANCY_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "tensor::scalar_algebra::tests::globally_preoccupied_alias_assumptions_and_hooks_are_never_adopted",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        let occupied = symbol!("idenso::scalar_algebra_coefficient_0"; Real, Positive);
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = calls.clone();
        let hooked = symbol!(
            "idenso::scalar_algebra_coefficient_1",
            norm = move |_, _| {
                observed.fetch_add(1, Ordering::SeqCst);
            }
        );
        let p = pair("global_occupancy_p", 8);
        let variable = symbol!("scalar_algebra::global_occupancy_t"; Scalar, Real, Positive);
        let marked = p.scalar_mul(&Atom::var(variable).pow(2)).unwrap();
        let (input, expansion) = series(&marked, variable, 2);
        assert!(input.get_aliases().keys().all(|alias| {
            !alias.get_all_symbols(true).contains(&occupied)
                && !alias.get_all_symbols(true).contains(&hooked)
        }));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_equal(&coefficient(&marked, &input, &expansion, 2), &p);
    }

    #[test]
    fn native_word_roots_keep_exterior_signs_and_are_never_commuting_aliases() {
        test_initialize();
        let head = symbol!("scalar_algebra::exterior"; Flat, Linear, Antisymmetric; tag=&SPENSO_TAG.tensor_word);
        let a = Tensor::infer(function!(
            SPENSO_TAG.tensor_symbol("scalar_algebra::ghost_a")
        ))
        .unwrap();
        let b = Tensor::infer(function!(
            SPENSO_TAG.tensor_symbol("scalar_algebra::ghost_b")
        ))
        .unwrap();
        let ab = Tensor::word(head, &[a.clone(), b.clone()]).unwrap();
        let ba = Tensor::word(head, &[b, a.clone()]).unwrap();
        assert_eq!(ab.expression(), &(-ba.expression()));
        assert!(Tensor::word(head, &[a.clone(), a]).unwrap().is_zero());
        let variable = symbol!("scalar_algebra::word_t"; Scalar, Real, Positive);
        let marked = ab.scalar_mul(&Atom::var(variable).pow(2)).unwrap();
        let (input, expansion) = series(&marked, variable, 2);
        assert!(input.get_aliases().is_empty());
        let restored = coefficient(&marked, &input, &expansion, 2);
        assert_eq!(restored.expression(), ab.expression());
        let nonlinear = ab.with_checked_expression(ab.expression().cos()).unwrap();
        assert!(
            nonlinear
                .scalar_algebra_input(variable, |_| true)
                .unwrap_err()
                .to_string()
                .contains("native tensor words")
        );
    }

    fn power_position(position: ScalarPayloadPosition<'_>) -> bool {
        matches!(
            position,
            ScalarPayloadPosition::PowerBase { .. } | ScalarPayloadPosition::PowerExponent { .. }
        )
    }

    fn checked_sum(values: &[Tensor]) -> Tensor {
        Tensor::infer(Atom::add_many(values.iter().map(Tensor::expression))).unwrap()
    }

    #[test]
    fn default_expansion_preserves_closed_nonselfdual_sum_powers() {
        let p = pair("default_expand_p", 101);
        let q = pair("default_expand_q", 103);
        let sum = checked_sum(&[p.clone(), q.clone()]);
        let square = checked_sum(&[
            p.pow(&constant(2)).unwrap(),
            p.multiply(&q).unwrap().scalar_mul(&Atom::num(2)).unwrap(),
            q.pow(&constant(2)).unwrap(),
        ]);
        let cube = checked_sum(&[
            p.pow(&constant(3)).unwrap(),
            p.pow(&constant(2))
                .unwrap()
                .multiply(&q)
                .unwrap()
                .scalar_mul(&Atom::num(3))
                .unwrap(),
            p.multiply(&q.pow(&constant(2)).unwrap())
                .unwrap()
                .scalar_mul(&Atom::num(3))
                .unwrap(),
            q.pow(&constant(3)).unwrap(),
        ]);
        for (order, expected) in [(2, square), (3, cube)] {
            let source = sum.pow(&constant(order)).unwrap();
            let before = source.clone();
            let expanded = source.expanded(None, false).unwrap();
            assert_equal(&expanded, &expected);
            assert_eq!(source, before);
            assert_eq!(expanded.expanded(None, false).unwrap(), expanded);
        }
        let difference = checked_sum(&[p.clone(), q.scalar_mul(&Atom::num(-1)).unwrap()]);
        let source = sum.multiply(&difference).unwrap();
        let expected = checked_sum(&[
            p.pow(&constant(2)).unwrap(),
            q.pow(&constant(2))
                .unwrap()
                .scalar_mul(&Atom::num(-1))
                .unwrap(),
        ]);
        assert_equal(&source.expanded(None, false).unwrap(), &expected);
    }

    #[test]
    fn default_expansion_preserves_other_representations_complex_weights_and_open_ports() {
        use crate::representations::SpinFundamental;
        let p = pair("default_expand_color", 107);
        let slot = SpinFundamental {}.to_symbolic([Atom::num(2), Atom::num(109)]);
        let q = Tensor::infer(
            function!(
                SPENSO_TAG.tensor_symbol("scalar_algebra::default_expand_spin_bar"),
                spenso::dind!(slot.clone())
            ) * function!(
                SPENSO_TAG.tensor_symbol("scalar_algebra::default_expand_spin"),
                slot
            ),
        )
        .unwrap();
        let weighted = q.scalar_mul(&Atom::i()).unwrap();
        let source = checked_sum(&[p.clone(), weighted])
            .pow(&constant(2))
            .unwrap();
        let terms = [
            p.pow(&constant(2)).unwrap(),
            p.multiply(&q)
                .unwrap()
                .scalar_mul(&(Atom::num(2) * Atom::i()))
                .unwrap(),
            q.pow(&constant(2))
                .unwrap()
                .scalar_mul(&Atom::num(-1))
                .unwrap(),
        ];
        let open = Tensor::infer(function!(
            SPENSO_TAG.tensor_symbol("scalar_algebra::default_expand_spectator"),
            ColorFundamental {}.to_symbolic([Atom::num(3), Atom::num(113)]),
            spenso::mink!(4, 127),
            spenso::mink!(4, 131)
        ))
        .unwrap();
        let source = source.multiply(&open).unwrap();
        let expected = checked_sum(
            &terms
                .iter()
                .map(|term| term.multiply(&open).unwrap())
                .collect::<Vec<_>>(),
        );
        let expanded = source.expanded(None, false).unwrap();
        assert_eq!(
            expanded.structure().logical_slots(),
            source.structure().logical_slots()
        );
        assert_equal(&expanded, &expected);
        let structure = PartialStructure::from_logical_slots(
            open.structure().logical_slots().into_iter().rev(),
        );
        let zero = Tensor::from_declared_parts(Atom::Zero, structure).unwrap();
        let expanded = zero.expanded(None, false).unwrap();
        assert_eq!(expanded.expression(), &Atom::Zero);
        assert_eq!(expanded.structure(), zero.structure());
    }

    #[test]
    fn default_expansion_does_not_enter_function_arguments_or_scalar_metadata() {
        let p = pair("default_expand_metadata_p", 137);
        let q = pair("default_expand_metadata_q", 139);
        let sum = checked_sum(&[p.clone(), q.clone()]);
        let square = sum.pow(&constant(2)).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = calls.clone();
        let head = symbol!("scalar_algebra::default_expand_metadata"; Scalar; norm=move |_, _| {
            observed.fetch_add(1, Ordering::SeqCst);
        });
        let metadata = function!(head, p.expression(), square.expression());
        let source = square.scalar_mul(&metadata).unwrap();
        let before = calls.load(Ordering::SeqCst);
        let expanded = source.expanded(None, false).unwrap();
        assert!(expanded.expression().contains(&metadata));
        assert_eq!(calls.load(Ordering::SeqCst), before);
        let expected = checked_sum(&[
            p.pow(&constant(2)).unwrap().scalar_mul(&metadata).unwrap(),
            p.multiply(&q)
                .unwrap()
                .scalar_mul(&(Atom::num(2) * &metadata))
                .unwrap(),
            q.pow(&constant(2)).unwrap().scalar_mul(&metadata).unwrap(),
        ]);
        assert_eq!(expanded.structure(), expected.structure());
        // The complete declared Scalar metadata is one exact indeterminate.
        // Symbolica extracts its coefficient without entering function arguments;
        // only that metadata-free mathematical tensor is alpha-canonicalized.
        let canonical_coefficient = |value: &Tensor| {
            assert_eq!(value.structure(), expanded.structure());
            assert!(value.expression().contains(&metadata));
            assert_eq!(calls.load(Ordering::SeqCst), before);
            let rows = value
                .expression()
                .coefficient_list::<i32>(std::slice::from_ref(&metadata));
            assert_eq!(rows.len(), 1);
            let (key, coefficient) = &rows[0];
            assert_eq!(key, &metadata);
            assert!(!coefficient.contains(&metadata));
            let mathematical =
                Tensor::from_declared_parts(coefficient.clone(), value.structure().clone())
                    .unwrap();
            let canonical = mathematical.canonize().unwrap();
            Tensor::from_declared_parts(
                canonical.expression().clone(),
                canonical.structure().clone(),
            )
            .unwrap();
            assert_eq!(canonical.structure(), value.structure());
            assert!(value.expression().contains(&metadata));
            assert_eq!(calls.load(Ordering::SeqCst), before);
            canonical
        };
        assert!(canonical_coefficient(&expanded).same_value(&canonical_coefficient(&expected)));
        assert_eq!(calls.load(Ordering::SeqCst), before);
        let opaque = Tensor::infer(function!(
            symbol!("scalar_algebra::default_expand_arbitrary_function"; Scalar),
            square.expression()
        ))
        .unwrap();
        assert_eq!(opaque.expanded(None, false).unwrap(), opaque);
    }

    #[test]
    fn default_expansion_keeps_native_words_and_nonlinear_word_refusals() {
        test_initialize();
        let head = symbol!("scalar_algebra::default_expand_exterior"; Flat, Linear, Antisymmetric; tag=&SPENSO_TAG.tensor_word);
        let a = Tensor::infer(function!(
            SPENSO_TAG.tensor_symbol("scalar_algebra::default_expand_ghost_a")
        ))
        .unwrap();
        let b = Tensor::infer(function!(
            SPENSO_TAG.tensor_symbol("scalar_algebra::default_expand_ghost_b")
        ))
        .unwrap();
        let word = Tensor::word(head, &[a.clone(), b.clone()]).unwrap();
        let x = Atom::var(symbol!("scalar_algebra::default_expand_x"));
        let y = Atom::var(symbol!("scalar_algebra::default_expand_y"));
        let coefficient = (&x + &y).pow(2);
        let source = word.scalar_mul(&coefficient).unwrap();
        let expected = checked_sum(&[
            word.scalar_mul(&x.pow(2)).unwrap(),
            word.scalar_mul(&(Atom::num(2) * &x * &y)).unwrap(),
            word.scalar_mul(&y.pow(2)).unwrap(),
        ]);
        let expanded = source.expanded(None, false).unwrap();
        assert_eq!(expanded.expression(), expected.expression());
        assert_eq!(expanded.structure(), expected.structure());
        assert_eq!(word.expanded(None, false).unwrap(), word);
        let open_a = Tensor::infer(function!(
            SPENSO_TAG.tensor_symbol("scalar_algebra::default_expand_open_ghost"),
            spenso::mink!(4, 149)
        ))
        .unwrap();
        let open_word = Tensor::word(head, &[open_a, b]).unwrap();
        let source = open_word.scalar_mul(&coefficient).unwrap();
        let expanded = source.expanded(None, false).unwrap();
        let expected = checked_sum(&[
            open_word.scalar_mul(&x.pow(2)).unwrap(),
            open_word.scalar_mul(&(Atom::num(2) * &x * &y)).unwrap(),
            open_word.scalar_mul(&y.pow(2)).unwrap(),
        ]);
        assert_eq!(expanded.expression(), expected.expression());
        assert_eq!(expanded.structure(), source.structure());
        let nonlinear = word
            .with_checked_expression(word.expression().cos())
            .unwrap();
        assert_eq!(nonlinear.expanded(None, false).unwrap(), nonlinear);
        assert!(
            nonlinear
                .prepare_scalar_algebra_input(None, |_| true)
                .unwrap_err()
                .to_string()
                .contains("native tensor words")
        );
    }

    #[test]
    fn expansion_without_a_variable_retains_registry_and_alias_leak_guards() {
        let p = pair("default_expand_registry_p", 151);
        let q = pair("default_expand_registry_q", 157);
        let source = checked_sum(&[p, q]).pow(&constant(2)).unwrap();
        let prepare = || {
            source
                .prepare_scalar_algebra_input(None, power_position)
                .unwrap()
        };
        let first = prepare();
        let second = prepare();
        assert!(!first.get_aliases().is_empty());
        assert!(
            first
                .get_aliases()
                .keys()
                .all(|alias| !second.get_aliases().contains_key(alias))
        );
        let original_symbols = source.expression().get_all_symbols(true);
        let alias_symbols = first
            .get_aliases()
            .keys()
            .flat_map(|alias| alias.get_all_symbols(true))
            .collect::<HashSet<_>>();
        assert!(
            first.get_root().get_all_symbols(true).iter().all(|symbol| {
                original_symbols.contains(symbol) || alias_symbols.contains(symbol)
            })
        );
        let result = source
            .with_scalar_algebra_result(&first, first.get_root().expand(), power_position)
            .unwrap();
        assert!(
            result
                .expression()
                .get_all_symbols(true)
                .iter()
                .all(|symbol| !alias_symbols.contains(symbol))
        );
        let alias = first.get_aliases().keys().next().unwrap();
        assert!(
            source
                .with_scalar_algebra_result(&first, pack(alias), power_position)
                .unwrap_err()
                .to_string()
                .contains("temporary alias")
        );
    }

    #[test]
    fn default_expansion_retains_independent_additive_dummy_scopes() {
        let p = pair("default_expand_shared_label_p", 163);
        let q = pair("default_expand_shared_label_q", 163);
        let source = checked_sum(&[p.clone(), q.clone()])
            .pow(&constant(2))
            .unwrap();
        // The input retains the same written dummy in each additive branch.
        // Its analytic expected product is constructed with explicit independent
        // native scopes; ordinary explicit operands do not silently freshen.
        assert!(matches!(
            p.multiply(&q),
            Err(
                super::super::composition::TensorCompositionError::InvalidExplicitMultiplicity { .. }
            )
        ));
        let expected_p = p
            .wrap_indices(symbol!("scalar_algebra::expected_p_scope"), true)
            .unwrap();
        let expected_q = q
            .wrap_indices(symbol!("scalar_algebra::expected_q_scope"), true)
            .unwrap();
        let expected = checked_sum(&[
            expected_p.pow(&constant(2)).unwrap(),
            expected_p
                .multiply(&expected_q)
                .unwrap()
                .scalar_mul(&Atom::num(2))
                .unwrap(),
            expected_q.pow(&constant(2)).unwrap(),
        ]);
        assert_equal(&source.expanded(None, false).unwrap(), &expected);
    }

    fn closed_conjugation_position(position: ScalarPayloadPosition<'_>) -> bool {
        mathematical_position(position)
            || matches!(position,
            ScalarPayloadPosition::FunctionArgument { function, argument: 0 }
                if [Symbol::CONJ, spenso::network::library::function_lib::INBUILTS.conj]
                    .contains(&function.get_symbol()))
    }

    #[test]
    fn closed_conjugation_series_preserves_complex_coefficients_duality_and_powers() {
        use spenso::{bracket, network::library::function_lib::INBUILTS};
        let p = pair("closed_conjugation_complex", 1093);
        let variable = symbol!("scalar_algebra::closed_conjugation_t"; Scalar, Real, Positive);
        let complex = Atom::var(symbol!("scalar_algebra::closed_conjugation_z"; Scalar));
        let marked = &complex * p.expression() * Atom::var(variable).pow(2);
        let source = Tensor::infer(INBUILTS.conj(bracket!(marked)))
            .unwrap()
            .pow(&constant(2))
            .unwrap();
        let expected = Tensor::infer(INBUILTS.conj(bracket!(p.expression())))
            .unwrap()
            .pow(&constant(2))
            .unwrap()
            .scalar_mul(&complex.conj().pow(2))
            .unwrap();
        let input = source
            .scalar_algebra_input(variable, closed_conjugation_position)
            .unwrap();
        let series = input
            .get_root()
            .series(variable, 0, SeriesDepth::absolute(4))
            .unwrap();
        assert!(series.coefficient(0.into()).unwrap().is_zero());
        let restored = source
            .with_scalar_algebra_result(
                &input,
                series.coefficient(4.into()).unwrap(),
                closed_conjugation_position,
            )
            .unwrap();
        assert_equal(&restored, &expected);
        assert_eq!(
            Tensor::infer(restored.expression().clone())
                .unwrap()
                .structure(),
            restored.structure()
        );
        assert!(!restored.expression().contains_symbol(variable));
        // A genuine complex coefficient retains its universal scalar CONJ.
        assert!(restored.expression().contains_symbol(Symbol::CONJ));
        let private_alias = input
            .get_aliases()
            .keys()
            .find(|alias| matches!(alias.as_view(), AtomView::Var(_)))
            .unwrap();
        assert!(
            source
                .with_scalar_algebra_result(
                    &input,
                    pack(private_alias),
                    closed_conjugation_position
                )
                .unwrap_err()
                .to_string()
                .contains("temporary alias")
        );
        let symbols = restored.expression().get_all_symbols(true);
        assert!(input.get_aliases().keys().all(|alias| {
            !matches!(alias.as_view(), AtomView::Var(variable)
                if symbols.contains(&variable.get_symbol()))
        }));
    }

    #[test]
    fn closed_conjugation_series_retains_open_spectators_and_word_refusals() {
        use spenso::{bracket, network::library::function_lib::INBUILTS};
        let p = pair("closed_conjugation_open", 1097);
        let variable = symbol!("scalar_algebra::closed_conjugation_open_t"; Scalar, Real, Positive);
        let closed =
            Tensor::infer(INBUILTS.conj(bracket!(p.expression() * Atom::var(variable).pow(2))))
                .unwrap();
        let open = Tensor::infer(function!(
            SPENSO_TAG.tensor_symbol("scalar_algebra::closed_conjugation_spectator"),
            ColorFundamental {}.to_symbolic([Atom::num(3), Atom::num(1103)])
        ))
        .unwrap();
        let source = closed.multiply(&open).unwrap();
        let expected = Tensor::infer(INBUILTS.conj(bracket!(p.expression())))
            .unwrap()
            .multiply(&open)
            .unwrap();
        let input = source
            .scalar_algebra_input(variable, closed_conjugation_position)
            .unwrap();
        let series = input
            .get_root()
            .series(variable, 0, SeriesDepth::absolute(2))
            .unwrap();
        let restored = source
            .with_scalar_algebra_result(
                &input,
                series.coefficient(2.into()).unwrap(),
                closed_conjugation_position,
            )
            .unwrap();
        assert_equal(&restored, &expected);
        assert_eq!(restored.structure(), source.structure());
        let zero = source
            .with_scalar_algebra_result(
                &input,
                series.coefficient(1.into()).unwrap(),
                closed_conjugation_position,
            )
            .unwrap();
        assert!(zero.is_zero());
        assert_eq!(zero.structure(), source.structure());
        let word_head = symbol!("scalar_algebra::closed_conjugation_word"; Flat, Linear, Antisymmetric; tag=&SPENSO_TAG.tensor_word);
        let a = Tensor::infer(function!(
            SPENSO_TAG.tensor_symbol("scalar_algebra::closed_conjugation_ghost_a")
        ))
        .unwrap();
        let b = Tensor::infer(function!(
            SPENSO_TAG.tensor_symbol("scalar_algebra::closed_conjugation_ghost_b")
        ))
        .unwrap();
        let word = Tensor::word(word_head, &[a, b]).unwrap();
        let nonlinear = Tensor::infer(INBUILTS.conj(word.expression())).unwrap();
        assert!(
            nonlinear
                .scalar_algebra_input(variable, closed_conjugation_position)
                .unwrap_err()
                .to_string()
                .contains("native tensor words")
        );
    }
}

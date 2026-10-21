use super::*;
use spenso::{
    network::tags::SPENSO_TAG,
    structure::{
        abstract_index::AbstractIndex,
        partial::{PartialIndex, PartialStructureExt},
        representation::{LibraryRep, Minkowski, RepName},
        slot::{DummyAind, IsAbstractSlot},
    },
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use symbolica::{atom::FunctionBuilder, function, id::ConditionResult, symbol};

type Tensor = SymbolicTensor<PartialStructure>;

fn initialize() {
    crate::test_support::test_initialize();
}

fn slot(label: usize) -> Atom {
    Minkowski {}
        .new_rep(4)
        .slot::<AbstractIndex, _>(AbstractIndex::Normal(label))
        .to_atom()
}

fn leaf(name: &str, arguments: &[Atom]) -> Tensor {
    Tensor::infer(
        FunctionBuilder::new(SPENSO_TAG.tensor_symbol(name))
            .add_args(arguments)
            .finish(),
    )
    .unwrap()
}

fn pair(name: &str, label: usize) -> Tensor {
    leaf(&format!("{name}_J"), &[slot(label)])
        .multiply(&leaf(&format!("{name}_K"), &[slot(label)]))
        .unwrap()
}

fn literal_rule(source: &Tensor, rhs: Atom) -> TensorRule<'static> {
    TensorRule::new(
        Pattern::Literal(source.expression.clone()),
        ReplaceWith::Pattern(Pattern::Literal(rhs).into()),
        None,
        0,
    )
    .unwrap()
}

fn canonical(value: &Tensor) -> Atom {
    value.canonize().unwrap().expression
}

fn scopes(value: &Tensor) -> HashSet<Symbol> {
    let mut scopes = HashSet::new();
    let observed =
        SlotMatcher::default().map_expression_indices(value.expression.as_view(), |slot| {
            if let AbstractIndex::Scoped(index) = slot.aind {
                scopes.insert(index.scope());
            }
            None::<AbstractIndex>
        });
    assert_eq!(observed, value.expression);
    scopes
}

#[test]
fn fresh_binding_opt_in_admits_new_pairs_and_preserves_legacy_refusal() {
    initialize();
    let source = leaf("fresh_tensor_rule::V", &[slot(11)]);
    let rhs = leaf("fresh_tensor_rule::K", &[slot(11)])
        .multiply(&pair("fresh_tensor_rule::minimal", 13))
        .unwrap();
    assert_eq!(
        source.structure.logical_slots(),
        rhs.structure.logical_slots()
    );
    let rule = literal_rule(&source, rhs.expression.clone());
    assert!(
        source
            .replace(&rule)
            .unwrap_err()
            .to_string()
            .contains("introduces explicit indices")
    );
    let actual = source
        .replace_rules_with_fresh_bindings(std::slice::from_ref(&rule))
        .unwrap();
    assert_eq!(
        actual.structure.logical_slots(),
        source.structure.logical_slots()
    );
    assert_eq!(canonical(&actual), canonical(&rhs));
    assert_eq!(scopes(&actual).len(), 1);
    // The old API's refusal is unchanged even after the opt-in application.
    assert!(source.replace(&rule).is_err());
}

#[test]
fn fresh_binding_keeps_ambient_spectator_ports_and_avoids_capture() {
    initialize();
    let variable = leaf("fresh_tensor_rule::ambient_V", &[slot(21)]);
    let spectator = leaf("fresh_tensor_rule::ambient_J", &[slot(23)]);
    let source = variable.multiply(&spectator).unwrap();
    let inserted = spectator
        .multiply(&leaf("fresh_tensor_rule::ambient_K", &[slot(23)]))
        .unwrap();
    let rhs = leaf("fresh_tensor_rule::ambient_W", &[slot(21)])
        .multiply(&inserted)
        .unwrap();
    let actual = source
        .replace_rules_with_fresh_bindings(&[literal_rule(&variable, rhs.expression.clone())])
        .unwrap();
    let expected = spectator
        .multiply(
            &rhs.wrap_indices(symbol!("fresh_tensor_rule::ambient_expected"), true)
                .unwrap(),
        )
        .unwrap();
    assert_eq!(
        actual.structure.logical_slots(),
        source.structure.logical_slots()
    );
    assert_eq!(canonical(&actual), canonical(&expected));
    assert_eq!(actual.structure.logical_slots().len(), 2);
}

#[test]
fn fresh_binding_preserves_full_ports_and_source_order_when_rhs_arguments_permute() {
    initialize();
    let source = leaf("fresh_tensor_rule::ordered_F", &[slot(31), slot(33)]);
    let rhs = leaf("fresh_tensor_rule::ordered_G", &[slot(33), slot(31)])
        .multiply(&pair("fresh_tensor_rule::ordered", 35))
        .unwrap();
    assert_ne!(
        source.structure.logical_slots(),
        rhs.structure.logical_slots()
    );
    assert!(InterfaceInference::additive_interfaces_match(
        &source.structure,
        &rhs.structure
    ));
    let actual = source
        .replace_rules_with_fresh_bindings(&[literal_rule(&source, rhs.expression.clone())])
        .unwrap();
    let expected = Tensor::from_declared_parts(rhs.expression, source.structure.clone()).unwrap();
    assert_eq!(
        actual.structure.logical_slots(),
        source.structure.logical_slots()
    );
    assert_eq!(canonical(&actual), canonical(&expected));
}

#[test]
fn fresh_binding_scopes_each_cached_insertion_and_runs_conditions_only_once() {
    initialize();
    let source_leaf = leaf("fresh_tensor_rule::cached_F", &[slot(41)]);
    let rhs = leaf("fresh_tensor_rule::cached_G", &[slot(41)])
        .multiply(&pair("fresh_tensor_rule::cached", 43))
        .unwrap();
    let x = Atom::var(symbol!("fresh_tensor_rule::cached_x"));
    let y = Atom::var(symbol!("fresh_tensor_rule::cached_y"));
    let source =
        Tensor::infer(&x * &source_leaf.expression + &y * &source_leaf.expression).unwrap();
    for (cache_size, expected_calls) in [(0, 2), (8, 1)] {
        let calls = Arc::new(AtomicUsize::new(0));
        let conditions = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&calls);
        let condition_seen = Arc::clone(&conditions);
        let raw = rhs.expression.clone();
        let rule = TensorRule::new(
            source_leaf.expression.to_pattern(),
            ReplaceWith::Map(Box::new(move |_| {
                seen.fetch_add(1, Ordering::Relaxed);
                raw.clone()
            })),
            Some(Condition::match_stack(move |_| {
                condition_seen.fetch_add(1, Ordering::Relaxed);
                ConditionResult::True
            })),
            cache_size,
        )
        .unwrap();
        let actual = source.replace_rules_with_fresh_bindings(&[rule]).unwrap();
        let expected = Tensor::infer(&x * &rhs.expression + &y * &rhs.expression).unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), expected_calls);
        assert_eq!(conditions.load(Ordering::Relaxed), 2);
        assert_eq!(
            scopes(&actual).len(),
            2,
            "cached values still need copy-local bindings"
        );
        assert_eq!(
            actual.structure.logical_slots(),
            source.structure.logical_slots()
        );
        assert_eq!(canonical(&actual), canonical(&expected));
    }
}

#[test]
fn fresh_binding_retains_original_priority_identity_and_does_not_revisit_rhs() {
    initialize();
    let f = leaf("fresh_tensor_rule::simultaneous_F", &[]);
    let g = leaf("fresh_tensor_rule::simultaneous_G", &[]);
    let h = leaf("fresh_tensor_rule::simultaneous_H", &[]);
    let source = f.try_add(&g).unwrap();
    let actual = source
        .replace_rules_with_fresh_bindings(&[
            literal_rule(&f, g.expression.clone()),
            literal_rule(&g, h.expression.clone()),
        ])
        .unwrap();
    assert!(actual.same_value(&g.try_add(&h).unwrap()));
    let identity = source
        .replace_rules_with_fresh_bindings(&[
            literal_rule(&f, f.expression.clone()),
            literal_rule(&f, h.expression.clone()),
        ])
        .unwrap();
    assert!(identity.same_value(&source));
    let absent = source
        .replace_rules_with_fresh_bindings(&[literal_rule(&h, Atom::Zero)])
        .unwrap();
    assert!(absent.same_value(&source));
    // Include an eligible same-head no-match entry, not just fixed-head pruning.
    let a = leaf("fresh_tensor_rule::diagonal", &[slot(51), slot(51)]);
    let b = leaf("fresh_tensor_rule::diagonal", &[slot(53), slot(53)]);
    let source = a.try_add(&b).unwrap();
    let actual = source
        .replace_rules_with_fresh_bindings(&[literal_rule(&a, h.expression.clone())])
        .unwrap();
    assert!(actual.same_value(&h.try_add(&b).unwrap()));
}

#[test]
fn fresh_binding_zero_open_and_invalid_source_guards_precede_callbacks() {
    initialize();
    let rep = Minkowski {}.new_rep(4).cast::<LibraryRep>();
    let head = SPENSO_TAG.tensor_symbol("fresh_tensor_rule::open");
    let open = Tensor::from_declared_parts(
        function!(head, rep.to_symbolic([])),
        PartialStructure::from_logical_slots([rep.slot(PartialIndex::open(0))]),
    )
    .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    let rule = TensorRule::new(
        open.expression.to_pattern(),
        ReplaceWith::Map(Box::new(move |_| {
            seen.fetch_add(1, Ordering::Relaxed);
            Atom::Zero
        })),
        None,
        0,
    )
    .unwrap();
    assert!(
        open.replace_rules_with_fresh_bindings(std::slice::from_ref(&rule))
            .is_err()
    );
    let zero = Tensor::from_declared_parts(Atom::Zero, open.structure.clone()).unwrap();
    let actual = zero
        .replace_rules_with_fresh_bindings(std::slice::from_ref(&rule))
        .unwrap();
    assert!(actual.same_value(&zero));
    let invalid = Tensor::new(
        function!(head, SPENSO_TAG.chain_in),
        PartialStructure::from_logical_slots([]),
    );
    assert!(invalid.replace_rules_with_fresh_bindings(&[rule]).is_err());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let source = leaf("fresh_tensor_rule::explicit_zero", &[slot(61), slot(63)]);
    let actual = source
        .replace_rules_with_fresh_bindings(&[literal_rule(&source, Atom::Zero)])
        .unwrap();
    assert!(actual.is_zero());
    assert_eq!(
        actual.structure.logical_slots(),
        source.structure.logical_slots()
    );
}

#[test]
fn fresh_binding_checks_complete_labels_representations_and_internal_multiplicity() {
    initialize();
    let source = leaf("fresh_tensor_rule::checked_F", &[slot(71)]);
    let good = leaf("fresh_tensor_rule::checked_G", &[slot(71)])
        .multiply(&pair("fresh_tensor_rule::checked", 73))
        .unwrap();
    assert!(
        source
            .replace_rules_with_fresh_bindings(&[literal_rule(&source, good.expression)])
            .is_ok()
    );
    let rep = LibraryRep::new_dual("fresh_tensor_rule::flavor")
        .unwrap()
        .new_rep(4);
    for wrong in [
        leaf("fresh_tensor_rule::wrong_label", &[slot(72)]).expression,
        leaf(
            "fresh_tensor_rule::wrong_dual_rep",
            &[rep.dual().slot(AbstractIndex::Normal(71)).to_atom()],
        )
        .expression,
        leaf(
            "fresh_tensor_rule::wrong_dimension",
            &[Minkowski {}
                .new_rep(5)
                .slot::<AbstractIndex, _>(AbstractIndex::Normal(71))
                .to_atom()],
        )
        .expression,
    ] {
        assert!(
            source
                .replace_rules_with_fresh_bindings(&[literal_rule(&source, wrong)])
                .is_err()
        );
    }
    // The inherited source proof does not admit ordinary non-self-dual ports.
    // This explicit refusal is not a successful wrong-variance test or a claim
    // that fresh-binding application already covers arbitrary representations.
    let unsupported = leaf(
        "fresh_tensor_rule::nonselfdual_source",
        &[rep.slot(AbstractIndex::Normal(71)).to_atom()],
    );
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    let rule = TensorRule::new(
        unsupported.expression.to_pattern(),
        ReplaceWith::Map(Box::new(move |_| {
            seen.fetch_add(1, Ordering::Relaxed);
            Atom::Zero
        })),
        None,
        0,
    )
    .unwrap();
    assert!(
        unsupported
            .replace_rules_with_fresh_bindings(&[rule])
            .is_err()
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let source = leaf("fresh_tensor_rule::bad_pair_F", &[]);
    let j = leaf("fresh_tensor_rule::bad_pair_J", &[slot(75)]);
    let malformed = j.expression.clone().pow(3);
    assert!(
        source
            .replace_rules_with_fresh_bindings(&[literal_rule(&source, malformed)])
            .is_err()
    );
    let mixed = Atom::one() + j.expression;
    assert!(
        source
            .replace_rules_with_fresh_bindings(&[literal_rule(&source, mixed)])
            .is_err()
    );
}

#[test]
fn fresh_binding_keeps_scalar_metadata_exact_and_does_not_open_math_powers_or_words() {
    initialize();
    let source_leaf = leaf("fresh_tensor_rule::opaque_F", &[slot(81)]);
    let metadata = function!(
        symbol!("fresh_tensor_rule::metadata"; Scalar),
        &source_leaf.expression
    );
    let source = Tensor::infer(&source_leaf.expression * &metadata).unwrap();
    let rhs = leaf("fresh_tensor_rule::opaque_G", &[slot(81)])
        .multiply(&pair("fresh_tensor_rule::opaque", 83))
        .unwrap();
    let actual = source
        .replace_rules_with_fresh_bindings(&[literal_rule(&source_leaf, rhs.expression.clone())])
        .unwrap();
    assert!(actual.expression.contains(metadata.as_view()));
    assert_eq!(
        actual.structure.logical_slots(),
        source.structure.logical_slots()
    );
    let expected = Tensor::infer(&metadata * &rhs.expression).unwrap();
    assert_eq!(canonical(&actual), canonical(&expected));

    let scalar = leaf("fresh_tensor_rule::hidden_scalar", &[]);
    let cosine = Tensor::infer(scalar.expression.cos()).unwrap();
    let rule = literal_rule(&scalar, Atom::Zero);
    assert!(
        cosine
            .replace_rules_with_fresh_bindings(std::slice::from_ref(&rule))
            .unwrap()
            .same_value(&cosine)
    );
    let word_head = symbol!("fresh_tensor_rule::opaque_word"; Flat, Linear, Antisymmetric; tag=&SPENSO_TAG.tensor_word);
    let word = Tensor::word(word_head, std::slice::from_ref(&scalar)).unwrap();
    assert!(word.replace_rules_with_fresh_bindings(&[rule]).is_err());
    let norm = source_leaf.multiply(&source_leaf).unwrap();
    let power = norm.pow(&Tensor::infer(Atom::num(2)).unwrap()).unwrap();
    assert!(
        power
            .replace_rules_with_fresh_bindings(&[literal_rule(&source_leaf, Atom::Zero)])
            .is_err()
    );
}

#[test]
fn fresh_binding_refuses_user_normalizers_without_replaying_them() {
    initialize();
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    let hooked = symbol!(
        "fresh_tensor_rule::hooked",
        tag = &SPENSO_TAG.tensor,
        norm = |_, _| {
            CALLS.fetch_add(1, Ordering::Relaxed);
        }
    );
    let source = leaf("fresh_tensor_rule::hook_F", &[slot(91)]);
    let rhs = function!(hooked, slot(91));
    let rule = literal_rule(&source, rhs.clone());
    let before = CALLS.load(Ordering::Relaxed);
    assert!(source.replace_rules_with_fresh_bindings(&[rule]).is_err());
    assert_eq!(CALLS.load(Ordering::Relaxed), before);
    let hooked_source = Tensor::infer(rhs).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    let rule = TensorRule::new(
        hooked_source.expression.to_pattern(),
        ReplaceWith::Map(Box::new(move |_| {
            seen.fetch_add(1, Ordering::Relaxed);
            Atom::Zero
        })),
        None,
        0,
    )
    .unwrap();
    let before = CALLS.load(Ordering::Relaxed);
    assert!(
        hooked_source
            .replace_rules_with_fresh_bindings(&[rule])
            .is_err()
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(CALLS.load(Ordering::Relaxed), before);
}

#[test]
fn fresh_binding_rechecks_changed_product_powers_without_stamping_rewrite_proofs() {
    initialize();
    let f = leaf("fresh_tensor_rule::power_F", &[slot(101)]);
    let g = leaf("fresh_tensor_rule::power_G", &[slot(101)]);
    let h = leaf("fresh_tensor_rule::power_H", &[slot(101)]);
    let source = f.multiply(&g).unwrap();
    let actual = source
        .replace_rules_with_fresh_bindings(&[
            literal_rule(&f, h.expression.clone()),
            literal_rule(&g, h.expression.clone()),
        ])
        .unwrap();
    let expected =
        Tensor::from_declared_parts(h.expression.clone().pow(2), source.structure.clone()).unwrap();
    assert_eq!(
        actual.structure.logical_slots(),
        source.structure.logical_slots()
    );
    assert_eq!(canonical(&actual), canonical(&expected));
    assert!(!actual.rewrites_preserve_interface());
    assert!(
        actual
            .replace_rules_with_fresh_bindings(&[literal_rule(&h, Atom::Zero)])
            .is_err()
    );
}

#[test]
fn fresh_binding_reserves_source_and_all_raw_rhs_scope_symbols_at_actual_frontier() {
    const CHILD: &str = "IDENSO_FRESH_TENSOR_FRONTIER_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("tensor::replacement::fresh_binding_tests::fresh_binding_reserves_source_and_all_raw_rhs_scope_symbols_at_actual_frontier")
            .arg("--test-threads=1")
            .env(CHILD, "1")
            .output().unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
        return;
    }
    initialize();
    let f = leaf("fresh_tensor_rule::frontier_F", &[]);
    let g = leaf("fresh_tensor_rule::frontier_G", &[]);
    let p = pair("fresh_tensor_rule::frontier", 111);
    let rule = literal_rule(&f, p.expression.clone());
    // Consume a genuine public native counter value, without parsing/resetting
    // it. Explicit-only construction after this point consumes no fresh index.
    let AbstractIndex::Dummy(previous) = AbstractIndex::new_dummy() else {
        unreachable!()
    };
    let occupied = AbstractIndex::Dummy(previous.checked_add(1).unwrap())
        .to_atom()
        .get_symbol()
        .unwrap();
    let ambient = p.wrap_indices(occupied, true).unwrap();
    let source = Tensor::infer(&ambient.expression * &f.expression).unwrap();
    let actual = source.replace_rules_with_fresh_bindings(&[rule]).unwrap();
    let independent = p
        .wrap_indices(symbol!("fresh_tensor_rule::frontier_expected"), true)
        .unwrap();
    let expected = ambient.multiply(&independent).unwrap();
    assert_eq!(
        actual.structure.logical_slots(),
        source.structure.logical_slots()
    );
    assert_eq!(canonical(&actual), canonical(&expected));

    let source = f.multiply(&g).unwrap();
    let q = pair("fresh_tensor_rule::frontier_second", 113);
    let AbstractIndex::Dummy(previous) = AbstractIndex::new_dummy() else {
        unreachable!()
    };
    let occupied = AbstractIndex::Dummy(previous.checked_add(1).unwrap())
        .to_atom()
        .get_symbol()
        .unwrap();
    // This scope symbol is absent from the complete source and appears only in
    // the later raw RHS's opaque metadata. It must still be reserved before
    // allocating a scope for the first RHS.
    let metadata = function!(
        symbol!("fresh_tensor_rule::frontier_raw_metadata"; Scalar),
        occupied
    );
    let later = Tensor::infer(&metadata * &q.expression).unwrap();
    let actual = source
        .replace_rules_with_fresh_bindings(&[
            literal_rule(&f, p.expression.clone()),
            literal_rule(&g, later.expression),
        ])
        .unwrap();
    assert!(actual.expression.contains(metadata.as_view()));
    assert!(!scopes(&actual).contains(&occupied));
    assert_eq!(scopes(&actual).len(), 2);
    let expected = Tensor::infer(&p.expression * &q.expression * &metadata).unwrap();
    assert_eq!(canonical(&actual), canonical(&expected));
}

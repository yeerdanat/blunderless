import math

from blunderless.stats.tests import benjamini_hochberg, two_proportion_z, wilson_ci


def test_z_test_known_value():
    # 40/200 vs 25/250: pooled p=65/450, se=sqrt(p(1-p)(1/200+1/250))
    # => z = (0.2-0.1)/0.033350 = 2.9985, two-sided p = erfc(z/sqrt2) ≈ 0.00271
    z, p = two_proportion_z(40, 200, 25, 250)
    assert math.isclose(z, 2.9985, abs_tol=1e-3)
    assert math.isclose(p, 0.00271, abs_tol=5e-5)


def test_z_test_no_difference():
    z, p = two_proportion_z(10, 100, 10, 100)
    assert z == 0.0
    assert p == 1.0


def test_z_test_empty_sides():
    assert two_proportion_z(0, 0, 5, 100) == (0.0, 1.0)


def test_bh_monotone_and_bounded():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    q = benjamini_hochberg(p)
    # classic worked example: q for the smallest p is p*m/1
    assert math.isclose(q[0], 0.01, abs_tol=1e-9)
    assert all(0 <= qi <= 1 for qi in q)
    # q respects p ordering
    ranked = sorted(zip(p, q, strict=True))
    for (_, q1), (_, q2) in zip(ranked, ranked[1:], strict=False):
        assert q1 <= q2


def test_bh_no_signal_stays_insignificant():
    # uniform-ish p-values: nothing should survive at q < 0.1
    p = [0.3, 0.5, 0.7, 0.2, 0.9, 0.45, 0.61, 0.8]
    q = benjamini_hochberg(p)
    assert min(q) > 0.1


def test_wilson_ci_contains_point_estimate():
    lo, hi = wilson_ci(20, 100)
    assert lo < 0.2 < hi
    assert wilson_ci(0, 0) == (0.0, 1.0)

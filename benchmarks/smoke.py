"""Fast, deterministic smoke test for the benchmark-facing WHR surface."""

from __future__ import annotations

import math

from whr import WHR


def main() -> None:
    model = WHR({"draw_rate": 0.25})
    results = ("B", "W", "B", "D", "B", "W", "D", "B")
    for day, result in enumerate(results, start=1):
        model.create_game(
            "home",
            "away",
            result,
            day,
            1.0,
        )

    model.auto_iterate(precision=1e-3)
    binary = model.probability_future_match(
        "home",
        "away",
        handicap_key=1.0,
        account_for_uncertainty=True,
    )
    ternary = model.win_draw_loss_probabilities(
        "home",
        "away",
        handicap_key=1.0,
        account_for_uncertainty=True,
    )

    assert all(math.isfinite(probability) for probability in (*binary, *ternary))
    assert math.isclose(sum(binary), 1.0)
    assert math.isclose(sum(ternary), 1.0)
    assert model.max_gradient_norm() <= 1e-3
    assert model.ratings_for_player("home")
    print("benchmark smoke: ok")


if __name__ == "__main__":
    main()

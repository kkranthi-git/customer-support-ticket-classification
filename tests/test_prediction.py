from src.predict import predict_ticket


def test_prediction_output():
    result = predict_ticket(
        subject="Payment failed",
        body="My payment is failing and I cannot complete the transaction."
    )

    assert isinstance(result, dict)
    assert "queue" in result
    assert "priority" in result


def test_prediction_values():
    result = predict_ticket(
        subject="Internet connection problem",
        body="My internet keeps disconnecting."
    )

    assert isinstance(result["queue"], str)
    assert isinstance(result["priority"], str)


def test_empty_ticket():
    result = predict_ticket(
        subject="",
        body=""
    )

    assert isinstance(result, dict)
    assert "queue" in result
    assert "priority" in result
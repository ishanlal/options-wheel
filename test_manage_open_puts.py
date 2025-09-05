import pytest
from unittest.mock import MagicMock, patch
from core.execution import manage_open_puts
from alpaca.trading.enums import AssetClass
from alpaca.trading.requests import MarketOrderRequest


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.trade_client.submit_order.return_value.id = "mock_order_id"
    return client


def make_mock_position(symbol, qty, avg_entry_price, asset_class=AssetClass.US_OPTION):
    return MagicMock(
        symbol=symbol,
        qty=qty,
        avg_entry_price=avg_entry_price,
        asset_class=asset_class
    )


def make_snapshot(price):
    latest_quote = MagicMock()
    latest_quote.bid_price = price * 0.9
    latest_quote.ask_price = price * 1.1

    latest_trade = MagicMock()
    latest_trade.price = price

    snapshot = MagicMock()
    snapshot.latest_quote = latest_quote
    snapshot.latest_trade = latest_trade
    return snapshot


def test_no_positions(mock_client, caplog):
    mock_client.get_positions.return_value = []
    result = manage_open_puts(mock_client)
    assert result is None
    assert "No open put positions to manage" in caplog.text


def test_only_calls_skipped(mock_client, caplog):
    mock_client.get_positions.return_value = [
        make_mock_position("AAPL250920C00150000", -1, 2.5)
    ]
    result = manage_open_puts(mock_client)
    assert result is None
    assert "No valid put positions found" in caplog.text


def test_invalid_option_symbol_skipped(mock_client, caplog):
    bad_symbol = "INVALID_SYMBOL"
    mock_client.get_positions.return_value = [
        make_mock_position(bad_symbol, -1, 2.0)
    ]
    result = manage_open_puts(mock_client)
    assert result is None
    assert f"Could not parse option symbol {bad_symbol}" in caplog.text


def test_put_below_threshold_not_closed(mock_client, caplog):
    mock_client.get_positions.return_value = [
        make_mock_position("AAPL250920P00150000", -1, 2.00)
    ]
    mock_client.get_stock_latest_trade.return_value = {
        "AAPL": MagicMock(price=149.00)
    }
    mock_client.get_option_snapshot.return_value = {
        "AAPL250920P00150000": make_snapshot(price=1.50)  # 25% P&L
    }

    result = manage_open_puts(mock_client, target_pct=0.90)
    assert result == []
    assert "P&L%" in caplog.text
    assert "Buying back" not in caplog.text


def test_put_hits_profit_target_and_closed(mock_client, caplog):
    mock_client.get_positions.return_value = [
        make_mock_position("AAPL250920P00150000", -1, 2.00)
    ]
    mock_client.get_stock_latest_trade.return_value = {
        "AAPL": MagicMock(price=155.00)
    }
    mock_client.get_option_snapshot.return_value = {
        "AAPL250920P00150000": make_snapshot(price=0.10)  # 95% profit
    }

    result = manage_open_puts(mock_client, target_pct=0.90)
    assert len(result) == 1
    assert result[0]['symbol'] == "AAPL250920P00150000"
    assert result[0]['reason'] == 'profit_target'
    assert "taking profit" in caplog.text


def test_put_hits_loss_limit_and_closed(mock_client, caplog):
    mock_client.get_positions.return_value = [
        make_mock_position("AAPL250920P00150000", -1, 2.00)
    ]
    mock_client.get_stock_latest_trade.return_value = {
        "AAPL": MagicMock(price=140.00)
    }
    mock_client.get_option_snapshot.return_value = {
        "AAPL250920P00150000": make_snapshot(price=3.90)  # -95% loss
    }

    result = manage_open_puts(mock_client, target_pct=0.90)
    assert len(result) == 1
    assert result[0]['reason'] == 'loss_limit'
    assert "cutting loss" in caplog.text


def test_missing_snapshot_skipped(mock_client, caplog):
    symbol = "AAPL250920P00150000"
    mock_client.get_positions.return_value = [
        make_mock_position(symbol, -1, 2.00)
    ]
    mock_client.get_stock_latest_trade.return_value = {
        "AAPL": MagicMock(price=150.00)
    }
    mock_client.get_option_snapshot.return_value = {}

    result = manage_open_puts(mock_client)
    assert result == []
    assert f"No option price data for {symbol}" in caplog.text


def test_submit_order_called_correctly(mock_client):
    mock_client.get_positions.return_value = [
        make_mock_position("AAPL250920P00150000", -2, 2.00)
    ]
    mock_client.get_stock_latest_trade.return_value = {
        "AAPL": MagicMock(price=155.00)
    }
    mock_client.get_option_snapshot.return_value = {
        "AAPL250920P00150000": make_snapshot(price=0.10)
    }

    manage_open_puts(mock_client, target_pct=0.90)

    # Check if submit_order was called with the correct request
    args, _ = mock_client.trade_client.submit_order.call_args
    order = args[0]
    assert isinstance(order, MarketOrderRequest)
    assert order.symbol == "AAPL250920P00150000"
    assert order.qty == 2
    assert order.side.name == "BUY"
    assert order.type.name == "MARKET"
    assert order.time_in_force.name == "DAY"

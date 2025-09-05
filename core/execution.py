import logging
from .strategy import filter_underlying, filter_options, score_options, select_options
from models.contract import Contract
import numpy as np
from .utils import parse_option_symbol
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, AssetClass

logger = logging.getLogger(f"strategy.{__name__}")

def sell_puts(client, allowed_symbols, buying_power, strat_logger = None):
    """
    Scan allowed symbols and sell short puts up to the buying power limit.
    """
    if not allowed_symbols or buying_power <= 0:
        return

    logger.info("Searching for put options...")
    filtered_symbols = filter_underlying(client, allowed_symbols, buying_power)
    strat_logger.set_filtered_symbols(filtered_symbols)
    if len(filtered_symbols) == 0:
        logger.info("No symbols found with sufficient buying power.")
        return
    option_contracts = client.get_options_contracts(filtered_symbols, 'put')
    snapshots = client.get_option_snapshot([c.symbol for c in option_contracts])
    put_options = filter_options([Contract.from_contract_snapshot(contract, snapshots.get(contract.symbol, None)) for contract in option_contracts if snapshots.get(contract.symbol, None)])
    if strat_logger:
        strat_logger.log_put_options([p.to_dict() for p in put_options])
    
    if put_options:
        logger.info("Scoring put options...")
        scores = score_options(put_options)
        put_options = select_options(put_options, scores)
        for p in put_options:
            buying_power -= 100 * p.strike 
            if buying_power < 0:
                break
            logger.info(f"Selling put: {p.symbol}")
            client.market_sell(p.symbol)
            if strat_logger:
                strat_logger.log_sold_puts([p.to_dict()])
    else:
        logger.info("No put options found with sufficient delta and open interest.")

def sell_calls(client, symbol, purchase_price, stock_qty, strat_logger = None):
    """
    Select and sell covered calls.
    """
    if stock_qty < 100:
        msg = f"Not enough shares of {symbol} to cover short calls!  Only {stock_qty} shares are held and at least 100 are needed!"
        logger.error(msg)
        raise ValueError(msg)

    logger.info(f"Searching for call options on {symbol}...")
    call_options = filter_options([Contract.from_contract(option, client) for option in client.get_options_contracts([symbol], 'call')], purchase_price)
    if strat_logger:
        strat_logger.log_call_options([c.to_dict() for c in call_options])

    if call_options:
        scores = score_options(call_options)
        contract = call_options[np.argmax(scores)]
        logger.info(f"Selling call option: {contract.symbol}")
        client.market_sell(contract.symbol)
        if strat_logger:
            strat_logger.log_sold_calls(contract.to_dict())
    else:
        logger.info(f"No viable call options found for {symbol}")

def manage_open_puts(client, target_pct=0.90, strat_logger=None):
    """
    Check open put positions and buy them back when unrealized profit/loss reaches 
    +/- target_pct of maximum possible profit (premium collected).
    
    Args:
        client: BrokerClient instance
        target_pct: Close position when profit OR loss reaches this percentage of premium collected (default 90%)
        strat_logger: Strategy logger for tracking closures
    """
    positions = client.get_positions()
    put_positions = [p for p in positions if p.asset_class == AssetClass.US_OPTION and int(p.qty) < 0]
    
    if not put_positions:
        logger.info("No open put positions to manage")
        return
    
    logger.info(f"Managing {len(put_positions)} open put positions")
    
    # Get current stock prices for all underlyings
    underlyings = []
    put_details = {}
    
    for position in put_positions:
        try:
            underlying, option_type, strike_price = parse_option_symbol(position.symbol)
            if option_type == 'P':  # Only process puts
                underlyings.append(underlying)
                put_details[position.symbol] = {
                    'underlying': underlying,
                    'strike': strike_price,
                    'qty': abs(int(position.qty)),
                    'avg_entry_price': abs(float(position.avg_entry_price)),  # Premium collected (make positive)
                    'position': position
                }
        except ValueError as e:
            logger.warning(f"Could not parse option symbol {position.symbol}: {e}")
            continue
    
    if not underlyings:
        logger.info("No valid put positions found")
        return
    
    # Get current stock prices
    try:
        stock_prices = client.get_stock_latest_trade(list(set(underlyings)))
    except Exception as e:
        logger.error(f"Failed to get stock prices: {e}")
        return
    
    # Get current option prices
    option_symbols = list(put_details.keys())
    try:
        option_snapshots = client.get_option_snapshot(option_symbols)
    except Exception as e:
        logger.error(f"Failed to get option snapshots: {e}")
        return
    
    positions_to_close = []
    
    for option_symbol, details in put_details.items():
        underlying = details['underlying']
        strike = details['strike']
        premium_collected = details['avg_entry_price']
        qty = details['qty']
        
        # Get current stock price
        if underlying not in stock_prices:
            logger.warning(f"No stock price data for {underlying}")
            continue
        
        current_stock_price = stock_prices[underlying].price
        
        # Get current option price
        if option_symbol not in option_snapshots:
            logger.warning(f"No option price data for {option_symbol}")
            continue
        
        option_snapshot = option_snapshots[option_symbol]
        
        # Use mid price or latest trade price
        if hasattr(option_snapshot, 'latest_quote') and option_snapshot.latest_quote:
            if option_snapshot.latest_quote.bid_price and option_snapshot.latest_quote.ask_price:
                current_option_price = (option_snapshot.latest_quote.bid_price + option_snapshot.latest_quote.ask_price) / 2
            else:
                current_option_price = option_snapshot.latest_quote.ask_price or 0
        elif hasattr(option_snapshot, 'latest_trade') and option_snapshot.latest_trade:
            current_option_price = option_snapshot.latest_trade.price
        else:
            logger.warning(f"No price data available for {option_symbol}")
            continue
        
        # Calculate P&L percentage
        # For short puts: profit when option price decreases, loss when it increases
        # Unrealized P&L = Premium Collected - Current Option Price
        unrealized_pnl = premium_collected - current_option_price
        
        # Calculate P&L as percentage of premium collected
        if premium_collected > 0:
            pnl_percentage = unrealized_pnl / premium_collected
        else:
            pnl_percentage = 0
        
        # Check closure conditions
        # Close if profit reaches +90% OR loss reaches -90%
        should_close = abs(pnl_percentage) >= target_pct
        
        # Log current status
        logger.info(f"{option_symbol}: Stock=${current_stock_price:.2f}, Strike=${strike:.2f}, "
                   f"Premium=${premium_collected:.2f}, Current=${current_option_price:.2f}, "
                   f"P&L=${unrealized_pnl:.2f}, P&L%={pnl_percentage:.1%}")
        
        if should_close:
            if pnl_percentage >= target_pct:
                reason = 'profit_target'
                logger.info(f"Profit target reached for {option_symbol}: "
                           f"Profit {pnl_percentage:.1%} >= {target_pct:.1%}, P&L=${unrealized_pnl:.2f}")
            else:  # pnl_percentage <= -target_pct
                reason = 'loss_limit'
                logger.info(f"Loss limit reached for {option_symbol}: "
                           f"Loss {pnl_percentage:.1%} <= -{target_pct:.1%}, P&L=${unrealized_pnl:.2f}")
            
            positions_to_close.append((details, reason, unrealized_pnl))
    
    # Close positions that meet criteria
    closed_positions = []
    for details, reason, pnl in positions_to_close:
        option_symbol = details['position'].symbol
        qty = details['qty']
        
        try:
            action_desc = "taking profit" if reason == 'profit_target' else "cutting loss"
            logger.info(f"Buying back put {option_symbol} - {action_desc}. P&L: ${pnl:.2f}")
            
            # Create buy-to-close order (positive quantity to close short position)
            close_order = MarketOrderRequest(
                symbol=option_symbol,
                qty=qty,  # Positive quantity to close short position
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY
            )
            
            order_response = client.trade_client.submit_order(close_order)
            logger.info(f"Successfully submitted close order for {option_symbol}: {order_response.id}")
            
            closed_positions.append({
                'symbol': option_symbol,
                'underlying': details['underlying'],
                'strike': details['strike'],
                'reason': reason,
                'pnl': pnl,
                'premium_collected': details['avg_entry_price'],
                'order_id': order_response.id
            })
            
        except Exception as e:
            logger.error(f"Failed to close position {option_symbol}: {e}")
    
    # Log closed positions
    if closed_positions and strat_logger:
        strat_logger.log_closed_puts(closed_positions)
    
    logger.info(f"Closed {len(closed_positions)} put positions out of {len(put_positions)} total")
    
    return closed_positions
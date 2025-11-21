# Position Detection Diagnostic

## Issue

Bot shows: `Current position (from Binance): SHORT_S1_LONG_S2`  
But Binance dashboard shows: **No open positions**

## Fixes Applied

✅ **Deprecated datetime.utcnow() fixed** - replaced with `datetime.now(datetime.UTC)`  
✅ **Added debug logging** to see actual Binance position amounts

## Diagnosis Steps

### 1. Verify Binance Positions

Check Binance Futures web dashboard:
- Do ADAUSDT or AVAXUSDT show ANY position amount?
- Even tiny amounts (0.01, 0.001) count!
- Look for "Position Amount" column

### 2. Enable DEBUG Logging

Edit `.env`:
```bash
LOG_LEVEL=DEBUG
```

Restart bot and look for:
```
Position detection: ADAUSDT=XXX, AVAXUSDT=YYY (from Binance API)
```

### 3. Common Causes

**Likely scenarios:**
1. **Dust positions**: Tiny amounts (< 1.0) that weren't fully closed
2. **Rounding errors**: API shows 0.00001 which is > 0
3. **Binance lag**: Position closed but API hasn't updated

### 4. Manual Fix (if needed)

If tiny positions exist on Binance:
```python
# Use Binance UI to close manually, or
# Place reduceOnly order for exact amount
```

## Code Changes

- [`src/trading.py`](file:///Users/dpanelli/Projects/btc_mn_copulas/src/trading.py#L554-L562): Added debug log
- All `datetime.utcnow()` → `datetime.now(datetime.UTC)`

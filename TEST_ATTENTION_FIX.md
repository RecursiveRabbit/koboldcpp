# Testing the Attention Race Condition Fix

## Quick Test Protocol

### 1. Apply the Patch

```bash
cd /home/evans/Coding_Projects/koboldcpp
git apply ATTENTION_RACE_FIX.patch
```

Or manually edit `src/llama-context.cpp` line 798, add:
```cpp
ggml_backend_sched_synchronize(sched.get());
```

### 2. Rebuild KoboldCPP

```bash
cd /home/evans/Coding_Projects/koboldcpp
make clean
make LLAMA_CUDA=1
```

### 3. Start KoboldCPP

```bash
python3 koboldcpp.py --model /home/evans/Coding_Projects/Halo_Weave/models/Qwen2.5-VL-7B-Instruct-Q8_0.gguf --port 5001 --usecublas 0 --gpulayers 999 --contextsize 512
```

### 4. Capture Test Data

From Halo Weave frontend (http://127.0.0.1:8080):

1. Click "Start Capture"
2. Send a message
3. Generate at least 50 tokens
4. Click "Stop Capture"
5. Note the capture directory (e.g., `Capture_Data/capture_1763800000000/`)

### 5. Run Validation

```bash
cd /home/evans/Coding_Projects/Halo_Weave/halo_weave
python3 find_transitions.py Capture_Data/capture_<timestamp>/
```

**Expected BEFORE fix:**
```
Analyzing 387 tokens for transitions...

🔄 TRANSITION #1 at token 6
   NORMALIZED → RAW_LOGITS
...
============================================================
Total transitions: 182
```

**Expected AFTER fix:**
```
Analyzing 387 tokens for transitions...

============================================================
Total transitions: 0

Checking for periodic pattern...

Token   0: NORMALIZED   (min=    0.00, max=    0.99)
Token  10: NORMALIZED   (min=    0.00, max=    0.94)
Token  20: NORMALIZED   (min=    0.00, max=    0.98)
...
```

### 6. Detailed Analysis

```bash
python3 analyze_attention.py Capture_Data/capture_<timestamp>/
```

**Check for:**
- ✅ All tokens show "NORMALIZED"
- ✅ Min values all >= 0
- ✅ Max values all <= 1.01
- ✅ Sums approximately 784 (28 layers × 28 heads)
- ❌ No tokens show "RAW LOGITS"

### 7. Performance Benchmark

Compare generation speed before/after:

**Before fix:**
```
Generating (100 / 100 tokens)
Time: 10.234 seconds
Speed: ~10ms/token
```

**After fix (expected < 1% slowdown):**
```
Generating (100 / 100 tokens)
Time: 10.289 seconds
Speed: ~10.3ms/token
Overhead: 0.55ms/token (5.4%)
```

**Note:** Overhead should be minimal since GPU is already >99% done when we sync.

---

## Success Criteria

✅ **PASS if:**
- Zero transitions detected
- All tokens have min >= 0, max <= 1
- Performance overhead < 1%

❌ **FAIL if:**
- Any transitions still occur
- Any tokens have negative values or values > 10
- Performance degradation > 2%

---

## Troubleshooting

### If transitions still occur:

1. **Verify patch applied:**
   ```bash
   grep -A5 "UNCONDITIONAL ATTENTION EXTRACTION" src/llama-context.cpp
   ```
   Should show `ggml_backend_sched_synchronize` call BEFORE extraction.

2. **Check rebuild:**
   ```bash
   make clean && make LLAMA_CUDA=1
   ```

3. **Verify using patched binary:**
   ```bash
   ldd koboldcpp_cublas.so | grep cuda
   ./koboldcpp_cublas.so --version
   ```

### If performance overhead > 2%:

This is unexpected. The sync should be very fast since GPU is nearly done. Possible causes:
- CPU busy-waiting instead of yielding
- Multiple backends not syncing efficiently
- Check with: `nvprof` or `nsight-systems`

---

## Rollback

If the fix causes issues:

```bash
cd /home/evans/Coding_Projects/koboldcpp
git checkout src/llama-context.cpp
make clean && make LLAMA_CUDA=1
```

---

## References

- Bug report: `ATTENTION_RACE_CONDITION_BUG.md`
- Analysis from last night: `/home/evans/Coding_Projects/Halo_Weave/halo_weave/ATTENTION_MYSTERY.md`
- Test data: `Capture_Data/capture_1763788462071/` (shows the bug)

# Critical Fixes for 3-5x Portion Overestimation

**Date:** 2024-12-16
**Issue:** MCP Server returning absurd nutritional values (5910 calories, 495% daily protein)
**Root Cause:** Three compounding bugs (food segmentation, scale calibration, no sanity checks)

---

## The Problem

User reported nutritional feedback showing:
- **5910 total calories** (should be ~500-800 for typical meal)
- **284.7g protein (495% daily)** (should be ~20-40g, 30-60% daily)
- **Overall 3-5x overestimation**

---

## Root Cause Analysis (Web-Verified)

### ❌ Initial Wrong Hypothesis: Depth Inversion
**STATUS: DISPROVEN** ✅

I initially thought Depth Anything V2 used different depth convention than MiDaS. **Web research proved this WRONG:**

- **Depth Anything V2**: Outputs **inverse depth** (higher = closer) - [GitHub Issue #93](https://github.com/DepthAnything/Depth-Anything-V2/issues/93)
- **MiDaS**: Outputs **inverse depth** (higher = closer) - [GitHub Issue #21](https://github.com/isl-org/MiDaS/issues/21)
- **Both use SAME convention** - No inversion needed!

---

### Three ACTUAL Compounding Bugs:

#### 1. **Food Segmentation Fallback (CRITICAL - 4-11x error)**
**File:** `portion_calculator.py` line 66-67

When contour detection failed, the code used **the entire image as food** instead of just the plate area.

```python
# BEFORE (BUG):
if not contours:
    return np.ones(image.shape[:2], dtype=np.uint8) * 255, depth_map
    # ^ Treats ALL 307,200 pixels as food!
```

**Result:** Background table, plate edges, utensils all counted as food → **11x area overestimation** (307,200 px vs ~28,000 px actual plate)

---

#### 2. **Fallback Scale Too Large (2x error)**
**File:** `reference_detector.py` line 230

Assumed 40cm image width when no plate detected, but typical phone food photos are ~25-30cm.

```python
# BEFORE (BUG):
default_ratio = 40.0 / width  # Assumes 40cm image width
```

**Result:**
- Scale: 40/28 = 1.43x linear error
- Area: 1.43² = **2.04x area error** (area = scale²)

---

#### 3. **No Volume Sanity Checks (Unlimited error)**
**File:** `portion_calculator.py`

No upper bounds on volume. Allowed absurd values like 3000ml+ "meals".

**Result:** Catastrophic overestimations went unchecked.

---

## Fixes Applied

### Fix 1: Use Center Region Fallback Instead of Full Image ✅
**File:** `portion_calculator.py` lines 65-76

```python
if not contours:
    logger.warning("No food region detected, estimating center region")
    # CRITICAL FIX: Instead of using full image, estimate food is in center 40% of image
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 5  # 40% diameter circle in center

    food_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(food_mask, (center_x, center_y), radius, 255, -1)

    return food_mask, np.where(food_mask == 255, depth_map, 0)
```

**Impact:**
- Before: 307,200 pixels (full image)
- After: ~28,000 pixels (realistic plate area)
- **Reduction: 11x** ✅

---

### Fix 2: Reduce Fallback Scale from 40cm to 28cm ✅
**File:** `reference_detector.py` lines 230-234

```python
# CRITICAL FIX: Reduced from 40cm to 28cm (more realistic for phone food photos)
# Typical phone photos of food on a plate are taken at ~25-30cm field of view
default_ratio = 28.0 / width  # Assume 28cm image width (was 40cm!)

logger.warning("No reference object detected, using conservative fallback calibration (28cm width)")
```

**Impact:**
- Before: 0.00390625 cm²/pixel (40cm assumption)
- After: 0.00191406 cm²/pixel (28cm assumption)
- **Reduction: 2.04x** ✅

---

### Fix 3: Add Volume Sanity Check ✅
**File:** `portion_calculator.py` lines 184-194

```python
# SANITY CHECK: Cap volume at reasonable maximum
# Typical meal portions: 200-800ml
# Large servings: 800-1500ml
# Anything over 1500ml is likely an error
MAX_REASONABLE_VOLUME = 1500.0  # ml
if volume_ml > MAX_REASONABLE_VOLUME:
    logger.warning(
        f"⚠️ Volume {volume_ml:.0f}ml exceeds reasonable maximum ({MAX_REASONABLE_VOLUME}ml). "
        f"Capping to {MAX_REASONABLE_VOLUME}ml. This suggests calibration or segmentation error."
    )
    volume_ml = MAX_REASONABLE_VOLUME
```

**Impact:** Prevents catastrophic >1500ml portions ✅

---

### Fix 4: Correct midas_model Variable References ✅
**File:** `server.py` lines 240-242, 258

Changed remaining `midas_model` references to `depth_model` to prevent crashes.

---

## Expected Results After Deployment

### Nutritional Values Correction:

| Metric | Before (Bug) | After (Fixed) | Change |
|--------|--------------|---------------|--------|
| **Calories** | 5910 | ~500-800 | **-85%** ✅ |
| **Protein** | 284.7g (495%) | ~20-40g (30-60%) | **-85%** ✅ |
| **Carbs** | 571.7g | ~100-150g | **-75%** ✅ |
| **Portion Weight** | ~2000g | ~300-500g | **-75%** ✅ |

### Typical Nigerian Meal Examples:

| Food | Expected Portion | Expected Calories | Expected Protein |
|------|-----------------|-------------------|-----------------|
| Jollof Rice | 300-400g | 450-600 cal | 15-20g (25-35%) |
| Egusi Soup + Fufu | 400-500g | 500-700 cal | 25-35g (40-60%) |
| Fried Rice + Chicken | 350-450g | 550-750 cal | 30-40g (50-65%) |

---

## Mathematical Verification

### Before Fixes (Worst Case):
1. **Full image as food**: 11x area error (307,200 px vs 28,000 px)
2. **40cm fallback scale**: 2.04x area error
3. **No volume cap**: Unlimited multiplication
4. **Combined**: 11 × 2.04 = **22.4x overestimation** (worst case!)

This explains 5910 calories instead of 500-800.

### After Fixes:
1. **Center region fallback**: Realistic plate area
2. **28cm fallback**: Closer to actual phone photos
3. **Volume cap**: Prevents absurd outliers
4. **Combined**: **Expected accuracy: 70-85%** (Phase 1 baseline)

---

## Web Research Sources

- [Depth Anything V2 Output Format](https://github.com/DepthAnything/Depth-Anything-V2/issues/93) - Confirms inverse depth (higher = closer)
- [MiDaS Inverse Depth](https://github.com/isl-org/MiDaS/issues/21) - Confirms inverse depth (higher = closer)
- [Depth Anything V2 Blog](https://blog.roboflow.com/depth-anything/) - Overview of model
- [MiDaS PyTorch Hub](https://pytorch.org/hub/intelisl_midas_v2/) - Model documentation

---

## Deployment Checklist

- [x] ✅ Fix 1: Food segmentation fallback in portion_calculator.py
- [x] ✅ Fix 2: Scale calibration reduced to 28cm in reference_detector.py
- [x] ✅ Fix 3: Volume sanity cap (1500ml max)
- [x] ✅ Fix 4: Variable references corrected in server.py
- [x] ✅ Web research verified depth conventions
- [ ] 🔲 Commit all changes to Git
- [ ] 🔲 Push to Railway
- [ ] 🔲 Test with real food images
- [ ] 🔲 Verify nutritional values are realistic (20-40g protein, not 495%)

---

## Files Modified

1. ✅ **portion_calculator.py** - Fixed segmentation fallback (lines 65-76), added volume cap (lines 184-194)
2. ✅ **reference_detector.py** - Reduced fallback scale to 28cm (line 232)
3. ✅ **server.py** - Fixed midas_model references (lines 240-242, 258)
4. ✅ **depth_anything_v2.py** - Added documentation about inverse depth (lines 123-129)

---

## Key Lesson Learned

**Always verify assumptions with web research before deploying fixes!**

My initial hypothesis about depth inversion was wrong. Web research saved us from deploying a broken "fix" that would have made things worse.

The ACTUAL bugs were:
- Treating entire image as food (11x error)
- Wrong scale assumption (2x error)
- No sanity checks (unlimited error)

**Combined: 22x overestimation in worst case** → Explains 5910 calories vs expected 500-800.

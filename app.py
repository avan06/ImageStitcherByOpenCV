import os
import gc
import cv2
import time
import tempfile
import mimetypes
import traceback
import numpy as np
import gradio as gr

# --- Logging Helper ---
def log_and_print(message, current_log=""):
    """Prints a message to the console and appends it to the log string."""
    print(message) # Print to console
    return current_log + message + "\n" # Append to log string with newline

# --- Helper Function: Crop Image by Percentage ---
def crop_image_by_percent(image, crop_top_percent=0.0, crop_bottom_percent=0.0):
    """
    Crops the top and/or bottom portion of an image based on percentage.

    Args:
        image: The input image (NumPy array).
        crop_top_percent: Percentage of height to crop from the top (0-100).
        crop_bottom_percent: Percentage of height to crop from the bottom (0-100).

    Returns:
        The cropped image (NumPy array), or the original image if cropping is not needed
        or percentages are invalid. Returns None if the input image is invalid.
    """
    if image is None or image.size == 0:
        # print("Warning: Invalid input image to crop_image_by_percent.")
        return None # Return None for invalid input

    if crop_top_percent < 0 or crop_top_percent > 100 or \
         crop_bottom_percent < 0 or crop_bottom_percent > 100:
        print(f"Warning: Invalid crop percentages ({crop_top_percent}%, {crop_bottom_percent}%). Must be between 0 and 100. Skipping crop.")
        return image

    if crop_top_percent == 0 and crop_bottom_percent == 0:
        return image # No cropping needed

    if crop_top_percent + crop_bottom_percent >= 100:
        print(f"Warning: Total crop percentage ({crop_top_percent + crop_bottom_percent}%) is 100% or more. Skipping crop.")
        return image

    try:
        h, w = image.shape[:2]

        pixels_to_crop_top = int(h * crop_top_percent / 100.0)
        pixels_to_crop_bottom = int(h * crop_bottom_percent / 100.0)

        start_row = pixels_to_crop_top
        end_row = h - pixels_to_crop_bottom

        # Ensure indices are valid after calculation
        if start_row >= end_row or start_row < 0 or end_row > h:
             print(f"Warning: Invalid calculated crop rows (start={start_row}, end={end_row} for height={h}). Skipping crop.")
             return image

        cropped_image = image[start_row:end_row, :]
        # print(f"Debug: Cropped by percentage from {image.shape} to {cropped_image.shape}")
        return cropped_image
    except Exception as e:
        print(f"Unexpected error during percentage cropping: {e}. Returning original image.")
        traceback.print_exc()
        return image

# --- Helper Function: Crop Black Borders ---
def crop_black_borders(image, enable_cropping=True, strict_no_black_edges=False):
    """
    Crops black borders (or transparent borders for BGRA) from an image.

    Args:
        image: The input image (NumPy array).
        enable_cropping: If False, returns the original image.
        strict_no_black_edges: If True, iteratively removes any remaining single black/transparent
                                 pixel lines from the edges after the initial crop.

    Returns:
        The cropped image (NumPy array), or the original image if cropping is disabled.
        Returns None if the input is invalid or strict cropping removes everything.
    """
    if not enable_cropping:
        return image
    if image is None or image.size == 0:
        return None

    try:
        mask_coords_found = False
        coords = None

        # Check Alpha Channel first 
        if len(image.shape) == 3 and image.shape[2] == 4:
            # If BGRA, use the Alpha channel to find the bounding box
            alpha_channel = image[:, :, 3]
            coords = cv2.findNonZero(alpha_channel)
            if coords is not None:
                mask_coords_found = True

        # Fallback: Attempt grayscale conversion (for BGR)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                coords = cv2.findNonZero(gray)
                if coords is not None:
                    mask_coords_found = True
            except cv2.error as e_gray:
                # print(f"Note: cvtColor to GRAY failed ({e_gray}), trying mask method.")
                gray = None # Reset gray if conversion failed
        elif len(image.shape) == 2:
            gray = image # Already grayscale
            coords = cv2.findNonZero(gray)
            if coords is not None:
                mask_coords_found = True

        # Fallback: Use mask if grayscale failed or shape is unusual
        if not mask_coords_found:
            try:
                # Create a mask where any channel/value is > 0
                mask = np.any(image > 0, axis=-1) if len(image.shape) == 3 else (image > 0)
                coords = cv2.findNonZero(mask.astype(np.uint8))
                if coords is not None:
                    mask_coords_found = True
            except Exception as e_crop_fallback:
                # print(f"Could not create mask for cropping fallback: {e_crop_fallback}. Returning original.")
                return image # Cannot proceed if mask fails too

        if not mask_coords_found or coords is None:
            # print("Debug: No non-black pixels found via any method, returning original.")
            return image # Return original if all black or coords failed

        x, y, w, h = cv2.boundingRect(coords)
        if w <= 0 or h <= 0:
            # print(f"Debug: Invalid bounding rect ({w}x{h}), returning original.")
            return image

        # Initial crop based on bounding rectangle
        cropped_image = image[y:y+h, x:x+w]

        # --- START: Strict Edge Cropping Logic ---
        if strict_no_black_edges and cropped_image is not None and cropped_image.size > 0:
            # Iteratively remove black edges until none remain or image is empty
            initial_shape = cropped_image.shape
            iterations = 0
            MAX_ITERATIONS = max(initial_shape) # Safety break
            while iterations < MAX_ITERATIONS:
                iterations += 1
                # Re-check size in loop
                if cropped_image is None or cropped_image.size == 0:
                    # print("Debug: Strict cropping resulted in empty image.")
                    return None # Image got cropped away entirely

                # Check current edges
                h_cr, w_cr = cropped_image.shape[:2]
                if h_cr <= 1 or w_cr <= 1:
                    break 

                # Extract edges depending on channels
                if len(cropped_image.shape) == 3 and cropped_image.shape[2] == 4:
                     # Use Alpha for strict check
                     top_row = cropped_image[0, :, 3]
                     bottom_row = cropped_image[-1, :, 3]
                     left_col = cropped_image[:, 0, 3]
                     right_col = cropped_image[:, -1, 3]
                elif len(cropped_image.shape) == 3:
                     # Convert to gray
                     try:
                        gray_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                     except:
                         # print("Warning: Failed to convert to gray during strict crop, stopping strict loop.")
                         break # Stop if conversion fails
                     top_row = gray_cropped[0, :]
                     bottom_row = gray_cropped[-1, :]
                     left_col = gray_cropped[:, 0]
                     right_col = gray_cropped[:, -1]
                else:
                     # Grayscale/Single channel
                     top_row = cropped_image[0, :]
                     bottom_row = cropped_image[-1, :]
                     left_col = cropped_image[:, 0]
                     right_col = cropped_image[:, -1]

                top_has_void = np.any(top_row == 0)
                bottom_has_void = np.any(bottom_row == 0)
                left_has_void = np.any(left_col == 0)
                right_has_void = np.any(right_col == 0)
                
                # If no edges have void, we are done
                if not (top_has_void or bottom_has_void or left_has_void or right_has_void):
                    # print(f"Debug: Strict cropping finished after {iterations-1} adjustments.")
                    break # Exit the while loop

                # Adjust cropping
                y_start_new, y_end_new = 0, h_cr
                x_start_new, x_end_new = 0, w_cr

                if top_has_void:
                    y_start_new += 1
                if bottom_has_void:
                    y_end_new -= 1
                if left_has_void:
                    x_start_new += 1
                if right_has_void:
                    x_end_new -= 1
                    
                # Check if new bounds are valid before slicing
                if y_start_new < y_end_new and x_start_new < x_end_new:
                    cropped_image = cropped_image[y_start_new:y_end_new, x_start_new:x_end_new]
                else:
                    # print("Debug: Strict cropping bounds became invalid, stopping.")
                    cropped_image = None # Signal that cropping failed
                    break # Exit loop

            if iterations >= MAX_ITERATIONS:
                 print("Warning: Strict cropping reached max iterations, potential issue.")
            if cropped_image is not None and initial_shape != cropped_image.shape:
                 print(f"Info: Strict cropping adjusted size from {initial_shape} to {cropped_image.shape}")
        # --- END: Strict Edge Cropping Logic ---

        return cropped_image # Return the potentially strictly cropped image

    except cv2.error as e:
        print(f"OpenCV Error during black border cropping: {e}. Returning uncropped image.")
        return image
    except Exception as e:
        print(f"Unexpected error during black border cropping: {e}. Returning uncropped image.")
        traceback.print_exc()
        return image

# --- Helper Function: Multi-Band Blending ---
def multi_band_blending(img1, img2, mask, num_levels=5):
    # img1, img2: The two images to blend (float32). Can be 3 or 4 channels (BGR or BGRA).
    # mask: The blending mask (float32, 0 to 1 transition, full canvas size, representing weight for img1)
    # num_levels: Number of pyramid levels
    log_message = "" # Add local logging if needed

    # Ensure inputs are float32 (caller should ensure this, but double check)
    if img1.dtype != np.float32:
        img1 = img1.astype(np.float32)
    if img2.dtype != np.float32:
        img2 = img2.astype(np.float32)

    if mask.dtype != np.float32:
        log_message = log_and_print(f"Warning: Mask input to multi_band_blending was {mask.dtype}, converting to float32.\n", log_message)
        if mask.max() > 1: # Assuming uint8 if max > 1
            mask = mask.astype(np.float32) / 255.0
        else: # Assuming already float but maybe not float32
            mask = mask.astype(np.float32)

    # Handle Channel Expansion for Mask
    # Ensure mask has same number of channels as images
    num_channels = img1.shape[2] if len(img1.shape) == 3 else 1
    
    if len(mask.shape) == 2 and len(img1.shape) == 3:
        if num_channels == 4:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA) # Expand to 4 channels
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Expand to 3 channels
    elif len(mask.shape) == 3 and mask.shape[2] == 1 and len(img1.shape) == 3:
         if num_channels == 4:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
         else:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 1. Build Gaussian pyramids for img1, img2
    gp1 = [img1]
    gp2 = [img2]
    # Temporary list to store pyrDown results to avoid modifying list during iteration
    gp1_next = []
    gp2_next = []
    actual_levels = 0
    for i in range(num_levels):
        prev_h, prev_w = gp1[-1].shape[:2]
        if prev_h < 2 or prev_w < 2:
            log_message = log_and_print(f"Warning: Stopping image pyramid build at level {i} due to small size ({prev_h}x{prev_w}).\n", log_message)
            break # Stop building pyramids for images

        try:
            down1 = cv2.pyrDown(gp1[-1])
            down2 = cv2.pyrDown(gp2[-1])
            gp1_next.append(down1)
            gp2_next.append(down2)
            actual_levels += 1 # Increment count of successfully built levels
        except cv2.error as e_pyrdown:
            log_message = log_and_print(f"Error during pyrDown at level {i+1}: {e_pyrdown}. Stopping pyramid build.\n", log_message)
            break # Stop if pyrDown fails
        
    # Update the main lists after the loop
    gp1.extend(gp1_next); del gp1_next
    gp2.extend(gp2_next); del gp2_next
    gc.collect()
    
    # Adjust num_levels to the actual number built
    num_levels = actual_levels
    
    # If pyramid build failed completely or input was too small
    if num_levels == 0:
        log_message = log_and_print("Error: Cannot build any pyramid levels. Using simple weighted average.\n", log_message)
        blended_img = img1 * mask + img2 * (1.0 - mask)
        blended_img = np.clip(blended_img, 0, 255).astype(np.uint8)
        # print(log_message) # Optional: print warnings
        if 'gp1' in locals(): del gp1
        if 'gp2' in locals(): del gp2
        gc.collect()
        return blended_img # Fallback

    # 2. Build Laplacian pyramids for img1, img2
    # Smallest Gaussian level acts as base of Laplacian pyramid
    lp1 = [gp1[num_levels]]
    lp2 = [gp2[num_levels]]
    for i in range(num_levels, 0, -1):
        # Target size is the size of the *next larger* Gaussian level
        target_size = (gp1[i-1].shape[1], gp1[i-1].shape[0])
        # log_message = log_and_print(f"Using resize instead of pyrUp for Laplacian level {i}\n", log_message) # Optional log
        ge1 = cv2.resize(gp1[i], target_size, interpolation=cv2.INTER_LINEAR)
        ge2 = cv2.resize(gp2[i], target_size, interpolation=cv2.INTER_LINEAR)

        # Ensure dimensions match EXACTLY before subtraction
        # Sometimes pyrUp result might be 1 pixel off from the actual gp[i-1] size
        h_target, w_target = gp1[i-1].shape[:2]
        h_ge, w_ge = ge1.shape[:2]

        # Crop or pad ge1/ge2 to match gp1[i-1]/gp2[i-1] dimensions
        if ge1.shape[:2] != (h_target, w_target):
            #print(f"Level {i} pyrUp/resize shape mismatch: ge1={ge1.shape}, target={gp1[i-1].shape}. Adjusting ge1.")
            ge1_adj = np.zeros_like(gp1[i-1], dtype=ge1.dtype)
            copy_h = min(h_target, h_ge)
            copy_w = min(w_target, w_ge)
            ge1_adj[:copy_h, :copy_w] = ge1[:copy_h, :copy_w]
            ge1 = ge1_adj
            del ge1_adj

        if ge2.shape[:2] != (h_target, w_target):
            #print(f"Level {i} pyrUp/resize shape mismatch: ge2={ge2.shape}, target={gp2[i-1].shape}. Adjusting ge2.")
            ge2_adj = np.zeros_like(gp2[i-1], dtype=ge2.dtype)
            copy_h = min(h_target, ge2.shape[0]) # Use ge2.shape[0] here
            copy_w = min(w_target, ge2.shape[1]) # Use ge2.shape[1] here
            ge2_adj[:copy_h, :copy_w] = ge2[:copy_h, :copy_w]
            ge2 = ge2_adj
            del ge2_adj


        # Calculate Laplacian: Higher resolution Gaussian - Expanded lower resolution Gaussian
        laplacian1 = cv2.subtract(gp1[i-1], ge1)
        laplacian2 = cv2.subtract(gp2[i-1], ge2)
        lp1.append(laplacian1)
        lp2.append(laplacian2)
        del ge1, ge2, laplacian1, laplacian2
        gc.collect()
        
    # del gp1, gp2
    # gc.collect()

    # lp1/lp2 lists are now [SmallestGaussian, LapN, LapN-1, ..., Lap1] (N=num_levels)
    lp1.reverse() # Reverse to [Lap1, ..., LapN, SmallestGaussian]
    lp2.reverse()

    # 3. Build Gaussian pyramid for the mask
    gm = [mask]
    gm_next = []
    actual_mask_levels = 0
    for i in range(num_levels): # Build mask pyramid only up to the actual image levels
        prev_h, prev_w = gm[-1].shape[:2]
        if prev_h < 2 or prev_w < 2:
            log_message = log_and_print(f"Warning: Stopping mask pyramid build at level {i}.\n", log_message)
            # num_levels should already be adjusted, but ensure mask levels don't exceed
            break
        try:
            down_mask = cv2.pyrDown(gm[-1])
            gm_next.append(down_mask)
            actual_mask_levels += 1
        except cv2.error as e_pyrdown_mask:
            log_message = log_and_print(f"Error during mask pyrDown at level {i+1}: {e_pyrdown_mask}. Stopping mask pyramid build.\n", log_message)
            break

    gm.extend(gm_next); del gm_next
    gc.collect()

    # Ensure mask pyramid has the same number of levels as laplacian (+ base)
    if len(gm) != num_levels + 1:
        log_message = log_and_print(f"Error: Mask pyramid levels ({len(gm)}) does not match expected ({num_levels + 1}). Using simple average.\n", log_message)
        # Fallback if mask pyramid construction failed unexpectedly
        blended_img = img1 * mask + img2 * (1.0 - mask)
        blended_img = np.clip(blended_img, 0, 255).astype(np.uint8)
        if 'lp1' in locals(): del lp1
        if 'lp2' in locals(): del lp2
        if 'gm' in locals(): del gm
        gc.collect()
        return blended_img


    # 4. Blend Laplacian levels
    ls = [] # Blended Laplacian pyramid
    for i in range(num_levels): # Blend Lap1 to LapN
        lap1 = lp1[i]
        lap2 = lp2[i]
        mask_level = gm[i] # Use corresponding mask level (gm[0] for lp1[0]=Lap1, etc.)

        # Ensure mask shape matches laplacian shape for this level
        if mask_level.shape[:2] != lap1.shape[:2]:
            # print(f"Level {i} mask/lap shape mismatch: mask={mask_level.shape}, lap={lap1.shape}. Resizing mask.")
            mask_level = cv2.resize(mask_level, (lap1.shape[1], lap1.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            # Ensure channels match after resize
            if len(mask_level.shape) == 2 and len(lap1.shape) == 3:
                if num_channels == 4:
                    mask_level = cv2.cvtColor(mask_level, cv2.COLOR_GRAY2BGRA)
                else:
                    mask_level = cv2.cvtColor(mask_level, cv2.COLOR_GRAY2BGR)
            elif len(mask_level.shape) == 3 and mask_level.shape[2] == 1 and len(lap1.shape) == 3:
                if num_channels == 4:
                    mask_level = cv2.cvtColor(mask_level, cv2.COLOR_GRAY2BGRA)
                else:
                    mask_level = cv2.cvtColor(mask_level, cv2.COLOR_GRAY2BGR)
            # Clip mask just in case resize interpolation goes slightly out of [0,1]
            mask_level = np.clip(mask_level, 0.0, 1.0)


        # Blend: L = L1*Gm + L2*(1-Gm)
        blended_lap = lap1 * mask_level + lap2 * (1.0 - mask_level)
        ls.append(blended_lap)
        del lap1, lap2, mask_level, blended_lap
        gc.collect()

    # Blend the smallest Gaussian level (base of the pyramid)
    base1 = lp1[num_levels] # Smallest Gaussian stored at the end of reversed lp1
    base2 = lp2[num_levels]
    mask_base = gm[num_levels] # Use the smallest mask (corresponding to the smallest Gaussian level)
    if mask_base.shape[:2] != base1.shape[:2]:
        # print(f"Base level mask/base shape mismatch: mask={mask_base.shape}, base={base1.shape}. Resizing mask.")
        mask_base = cv2.resize(mask_base, (base1.shape[1], base1.shape[0]), interpolation=cv2.INTER_LINEAR)
        # Channel check
        if len(mask_base.shape) == 2 and len(base1.shape) == 3:
             if num_channels == 4:
                 mask_base = cv2.cvtColor(mask_base, cv2.COLOR_GRAY2BGRA)
             else:
                 mask_base = cv2.cvtColor(mask_base, cv2.COLOR_GRAY2BGR)
        elif len(mask_base.shape) == 3 and mask_base.shape[2]==1 and len(base1.shape) == 3:
             if num_channels == 4:
                 mask_base = cv2.cvtColor(mask_base, cv2.COLOR_GRAY2BGRA)
             else:
                 mask_base = cv2.cvtColor(mask_base, cv2.COLOR_GRAY2BGR)
        mask_base = np.clip(mask_base, 0.0, 1.0)

    # Blend the base Gaussian level: B = B1*Gm_N + B2*(1-Gm_N)
    blended_base = base1 * mask_base + base2 * (1.0 - mask_base)
    ls.append(blended_base) # ls is now [BlendedLap1, ..., BlendedLapN, BlendedBase]
    # del lp1, lp2, gm, base1, base2, mask_base, blended_base
    del base1, base2, mask_base, blended_base
    gc.collect()

    # 5. Reconstruct the final image from the blended Laplacian pyramid
    # Start with the smallest blended base
    blended_img = ls[num_levels]
    for i in range(num_levels - 1, -1, -1): # Iterate from N-1 down to 0
        # Target size is the size of the *current* blended Laplacian level (ls[i])
        target_size = (ls[i].shape[1], ls[i].shape[0])
        # log_message = log_and_print(f"Using resize instead of pyrUp for reconstruction level {i}\n", log_message) # Optional log
        expanded_prev = cv2.resize(blended_img, target_size, interpolation=cv2.INTER_LINEAR)

        # Delete previous level's blended_img (important for memory)
        del blended_img
        gc.collect()

        # Ensure dimensions match EXACTLY before adding
        h_target_rec, w_target_rec = ls[i].shape[:2]
        h_exp, w_exp = expanded_prev.shape[:2]
        if expanded_prev.shape[:2] != (h_target_rec, w_target_rec):
            # print(f"Reconstruction level {i} shape mismatch: expanded={expanded_prev.shape}, target={ls[i].shape}. Adjusting expanded.")
            expanded_adj = np.zeros_like(ls[i], dtype=expanded_prev.dtype)
            copy_h_rec = min(h_target_rec, h_exp)
            copy_w_rec = min(w_target_rec, w_exp)
            expanded_adj[:copy_h_rec, :copy_w_rec] = expanded_prev[:copy_h_rec, :copy_w_rec]
            expanded_prev = expanded_adj
            del expanded_adj

        # Add the blended Laplacian for the current level
        current_laplacian = ls[i] # Get reference before add
        blended_img = cv2.add(expanded_prev, current_laplacian)
        del expanded_prev, current_laplacian # Remove laplacian reference ls[i]
        ls[i] = None # Explicitly break the reference in the list too? Might help GC.
        gc.collect()

    # Clip final result and convert back to uint8
    blended_img = np.clip(blended_img, 0, 255)
    blended_img = blended_img.astype(np.uint8)

    # Optional: print warnings collected during the process
    # if log_message: print("MultiBand Blend Logs:\n" + log_message)

    # Cleanup intermediate pyramids (important for memory)
    del gp1, gp2, lp1, lp2, gm, ls
    if 'laplacian1' in locals(): del laplacian1
    if 'laplacian2' in locals(): del laplacian2
    if 'ge1' in locals(): del ge1
    if 'ge2' in locals(): del ge2
    if 'mask_level' in locals(): del mask_level
    if 'base1' in locals(): del base1
    if 'base2' in locals(): del base2
    if 'mask_base' in locals(): del mask_base
    if 'blended_lap' in locals(): del blended_lap
    if 'blended_base' in locals(): del blended_base
    if 'expanded_prev' in locals(): del expanded_prev
    gc.collect()

    return blended_img

# --- Stitching Function: Focus on the pairwise images ---
def stitch_pairwise_images(img_composite, img_new,
                                                         transform_model_str="Homography",
                                                         blend_method="multi-band",
                                                         enable_gain_compensation=True,
                                                         orb_nfeatures=2000,
                                                         match_ratio_thresh=0.75,
                                                         ransac_reproj_thresh=5.0,
                                                         max_distance_coeff=0.5,
                                                         max_blending_width=10000,
                                                         max_blending_height=10000,
                                                         blend_smooth_ksize=15,
                                                         num_blend_levels=4
                                                        ):
    """
    Stitches a new image (img_new) onto an existing composite image (img_composite)
    using an explicit, step-by-step pipeline (e.g., ORB features).
    Allows choosing the geometric transformation model.
    Returns the new composite.
    Now handles BGRA (4-channel) images to preserve transparency.
    """
    log_message = log_and_print("--- Starting pairwise stitch between composite and new image (BGRA Mode) ---\n", "")
    start_time_pairwise = time.time()

    # --- Input Validation ---
    if img_composite is None or img_new is None:
        log_message = log_and_print("Error: One or both input images are None for the pairwise stitching step.\n", log_message)
        return None, log_message
    if img_composite.size == 0 or img_new.size == 0:
        log_message = log_and_print("Error: One or both input images are empty for the pairwise stitching step.\n", log_message)
        return None, log_message

    # --- Helper: Ensure 4-Channel BGRA ---
    def ensure_bgra(img):
        if img.ndim == 2: # Grayscale
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3: # BGR
            return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return img # Already BGRA

    # Convert inputs to BGRA
    img_composite = ensure_bgra(img_composite)
    img_new = ensure_bgra(img_new)

    h1, w1 = img_composite.shape[:2]
    h2, w2 = img_new.shape[:2]
    log_message = log_and_print(f"Pairwise Stitch: Img1({w1}x{h1}), Img2({w2}x{h2})\n", log_message)
    log_message = log_and_print(f"Params: Transform={transform_model_str}, ORB Feats={orb_nfeatures}, Ratio Thresh={match_ratio_thresh}\n", log_message)
    log_message = log_and_print(f"Params Cont'd: RANSAC Thresh={ransac_reproj_thresh}, Max Distance Coeff={max_distance_coeff}\n", log_message)
    log_message = log_and_print(f"Blending: Method={blend_method}, GainComp={enable_gain_compensation}, SmoothKSize={blend_smooth_ksize}, MB Levels={num_blend_levels}\n", log_message)

    final_output_img = None # Initialize result variable
    # Initialize other variables to None for better cleanup management
    img1_u8, img2_u8 = None, None
    kp1, des1, kp2, des2 = None, None, None, None
    all_matches, good_matches = None, None
    src_pts, dst_pts = None, None
    H_matrix_3x3_for_canvas = None # Will hold the 3x3 matrix for canvas calculation (Affine or Homography)
    final_warp_M = None # Will hold the actual 2x3 or 3x3 matrix for warping
    mask_trans = None # Mask from estimation function (homography or affine)
    pts1, dst_pts1_transformed = None, None
    pts2, all_pts = None, None
    output_img = None
    warped_img1_u8 = None
    mask_warped, mask_img2, overlap_mask = None, None, None
    gain_applied_warped_img1_u8 = None
    output_img_before_mb_float, blend_mask_float = None, None
    img1_for_blend, img2_for_blend = None, None
    is_affine = False # Flag to determine warp function

    try:
        # --- Feature Detection and Matching ---
        img1_u8 = img_composite.clip(0, 255).astype(np.uint8) if img_composite.dtype != np.uint8 else img_composite
        img2_u8 = img_new.clip(0, 255).astype(np.uint8) if img_new.dtype != np.uint8 else img_new

        # Use Grayscale for Feature Detection (ORB doesn't need Alpha)
        img1_gray = cv2.cvtColor(img1_u8, cv2.COLOR_BGRA2GRAY)
        img2_gray = cv2.cvtColor(img2_u8, cv2.COLOR_BGRA2GRAY)

        orb = cv2.ORB_create(nfeatures=orb_nfeatures)
        kp1, des1 = orb.detectAndCompute(img1_gray, None) # keypoints and descriptors
        kp2, des2 = orb.detectAndCompute(img2_gray, None)
        
        # Cleanup gray images used for detection
        del img1_gray, img2_gray
        gc.collect()

        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            log_message = log_and_print("Error: Not enough keypoints or descriptors found.\n", log_message)
            if 'kp1' in locals(): del kp1
            if 'des1' in locals(): del des1
            if 'kp2' in locals(): del kp2
            if 'des2' in locals(): del des2
            del img1_u8, img2_u8
            gc.collect()
            return None, log_message
        log_message = log_and_print(f"Found {len(kp1)} keypoints in Img1, {len(kp2)} in Img2.\n", log_message)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        # Check if descriptors are suitable for knnMatch (should be if ORB)
        if des1.dtype != np.uint8:
            des1 = des1.astype(np.uint8)
        if des2.dtype != np.uint8:
            des2 = des2.astype(np.uint8)
        all_matches = bf.knnMatch(des1, des2, k=2)
        del des1, des2; des1, des2 = None, None # Explicit delete
        gc.collect()

        good_matches = []
        if all_matches is not None:
            MAX_DISTANCE = max_distance_coeff * np.sqrt(w1**2 + h1**2)
            # Filter out potential empty match pairs
            valid_matches = [pair for pair in all_matches if isinstance(pair, (list, tuple)) and len(pair) == 2]
            for m, n in valid_matches:
                if m.distance < match_ratio_thresh * n.distance:
                    src_pt = np.array(kp1[m.queryIdx].pt)
                    dst_pt = np.array(kp2[m.trainIdx].pt)
                    distance = np.linalg.norm(dst_pt - src_pt)

                    if distance < MAX_DISTANCE:
                        good_matches.append(m)
            del valid_matches
        del all_matches; all_matches = None
        gc.collect()

        log_message = log_and_print(f"Found {len(good_matches)} good matches after ratio test.\n", log_message)
        MIN_MATCH_COUNT = 10 # Keep a minimum threshold

        # --- Transformation Estimation (Homography or Affine) ---
        if len(good_matches) >= MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
            del kp1, kp2, good_matches; kp1, kp2, good_matches = None, None, None # Explicit delete
            gc.collect()

            estimation_failed = False
            # Try Affine if selected
            if transform_model_str == "Affine_Partial" or transform_model_str == "Affine_Full":
                is_affine = True # Assume success initially
                affine_matrix_2x3 = None
                mask_a = None
                try:
                    if transform_model_str == "Affine_Partial":
                        log_message = log_and_print(f"Attempting Affine Partial Estimation (RANSAC Thresh={ransac_reproj_thresh})...\n", log_message)
                        affine_matrix_2x3, mask_a = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_reproj_thresh)
                    else: # Affine_Full
                        log_message = log_and_print(f"Attempting Affine Full Estimation (RANSAC Thresh={ransac_reproj_thresh})...\n", log_message)
                        affine_matrix_2x3, mask_a = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_reproj_thresh)

                    if affine_matrix_2x3 is None:
                        raise ValueError(f"{transform_model_str} estimation returned None")

                    # Convert 2x3 affine to 3x3 for canvas calculation consistency
                    H_matrix_3x3_for_canvas = np.vstack([affine_matrix_2x3, [0, 0, 1]]).astype(np.float64)
                    final_warp_M = affine_matrix_2x3.astype(np.float64) # Keep 2x3 for warpAffine
                    mask_trans = mask_a # Store the mask

                except Exception as e_affine:
                    log_message = log_and_print(f"Error during {transform_model_str} estimation: {e_affine}. Falling back to Homography.\n", log_message)
                    is_affine = False # Reset flag, will proceed to Homography block below
                    estimation_failed = True # Mark that the chosen affine failed
                    # Clean up affine specific vars if they exist
                    if 'affine_matrix_2x3' in locals():
                        del affine_matrix_2x3
                    if 'mask_a' in locals():
                        del mask_a
                    H_matrix_3x3_for_canvas = None
                    final_warp_M = None
                    mask_trans = None
                    # NOTE: We are choosing to fall back instead of returning None immediately.
                    # If you prefer to fail hard if the selected affine fails, uncomment the next line:
                    # return None, log_message
            
            # Try Homography if selected OR if Affine failed and we are falling back
            if not is_affine or estimation_failed: # If Homography was chosen or Affine failed
                if estimation_failed: # Log if we are falling back
                    log_message = log_and_print("Falling back to Homography estimation...\n", log_message)
                else: # Log if Homography was the original choice
                    log_message = log_and_print("Attempting Homography Estimation...\n", log_message)

                is_affine = False # Ensure flag is False for Homography path
                H_matrix_homog = None
                mask_h = None
                try:
                    log_message = log_and_print(f"Estimating Homography (RANSAC Thresh={ransac_reproj_thresh})...\n", log_message)
                    H_matrix_homog, mask_h = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=ransac_reproj_thresh)
                    if H_matrix_homog is None:
                        raise ValueError("Homography estimation returned None")

                    H_matrix_3x3_for_canvas = H_matrix_homog.astype(np.float64) # Use this for canvas calc
                    final_warp_M = H_matrix_homog.astype(np.float64) # Use 3x3 for warpPerspective
                    mask_trans = mask_h # Store the mask

                except Exception as e_homog:
                    log_message = log_and_print(f"Error during Homography estimation: {e_homog}\n", log_message)
                    # Clean up if Homography itself fails
                    if 'H_matrix_homog' in locals():
                        del H_matrix_homog
                    if 'mask_h' in locals():
                        del mask_h
                    del src_pts, dst_pts
                    gc.collect()
                    return None, log_message # Fail if Homography (chosen or fallback) fails

            # --- Log Inliers from the successful estimation ---
            model_name = "Affine" if is_affine else "Homography"
            if mask_trans is not None:
                inlier_count = np.sum(mask_trans)
                log_message = log_and_print(f"{model_name} estimated with {inlier_count} inliers.\n", log_message)
                if inlier_count < MIN_MATCH_COUNT:
                    log_message = log_and_print(f"Warning: Inlier count ({inlier_count}) < MIN_MATCH_COUNT for {model_name}. Result might be poor.\n", log_message)
                del mask_trans; mask_trans = None # Delete the mask now
                gc.collect()
            else:
                log_message = log_and_print(f"Warning: {model_name} mask was None.\n", log_message)


            # --- Cleanup source/destination points ---
            del src_pts, dst_pts; src_pts, dst_pts = None, None
            gc.collect()

            # --- Canvas Calculation and Warping ---
            pts1 = np.float32([[0,0],[0,h1-1],[w1-1,h1-1],[w1-1,0]]).reshape(-1,1,2)
            try:
                # Use the 3x3 matrix (derived from affine or directly from homography) for perspectiveTransform
                # Ensure it's float64
                if H_matrix_3x3_for_canvas.dtype != np.float64:
                    H_matrix_3x3_for_canvas = H_matrix_3x3_for_canvas.astype(np.float64)
                dst_pts1_transformed = cv2.perspectiveTransform(pts1, H_matrix_3x3_for_canvas)
                if dst_pts1_transformed is None:
                    raise ValueError("perspectiveTransform returned None")
            except Exception as e_tf:
                model_name_tf = "Affine-derived" if is_affine else "Homography"
                log_message = log_and_print(f"Error during perspectiveTransform (using {model_name_tf} 3x3 matrix): {e_tf}\n", log_message)
                # Clean up before returning
                del pts1
                if 'H_matrix_3x3_for_canvas' in locals(): del H_matrix_3x3_for_canvas
                if 'final_warp_M' in locals(): del final_warp_M # Was holding the warp matrix
                gc.collect()
                return None, log_message
            del pts1; pts1 = None

            pts2 = np.float32([[0,0],[0,h2-1],[w2-1,h2-1],[w2-1,0]]).reshape(-1,1,2)
            # Ensure dst_pts1_transformed is float32 for concatenation if needed
            all_pts = np.concatenate((pts2, dst_pts1_transformed.astype(np.float32)), axis=0)
            del pts2, dst_pts1_transformed; pts2, dst_pts1_transformed = None, None

            padding = 2
            x_min, y_min = np.int32(all_pts.min(axis=0).ravel() - padding)
            x_max, y_max = np.int32(all_pts.max(axis=0).ravel() + padding)
            del all_pts; all_pts = None
            gc.collect()

            translation_dist = [-x_min, -y_min]
            H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]], dtype=np.float64)

            output_width = x_max - x_min
            output_height = y_max - y_min

            if output_width <= 0 or output_height <= 0 or output_width > max_blending_width or output_height > max_blending_height:
                log_message = log_and_print(f"Error: Invalid output dimensions ({output_width}x{output_height}). Max allowed ({max_blending_width}x{max_blending_height})\n", log_message)
                # Clean up before returning
                if 'H_matrix_3x3_for_canvas' in locals():
                    del H_matrix_3x3_for_canvas
                if 'final_warp_M' in locals():
                    del final_warp_M
                if 'H_translation' in locals():
                    del H_translation
                gc.collect()
                return None, log_message

            log_message = log_and_print(f"Calculated canvas size: {output_width}x{output_height}\n", log_message)

            # --- Memory Check for Blending ---
            canvas_pixels = output_width * output_height
            # Define a threshold based on available memory, e.g., 250 million pixels
            # 15000*15000 = 225M, 30000*15000 = 450M
            pixel_threshold = 225_000_000
            effective_blend_method = blend_method

            if blend_method == "multi-band" and canvas_pixels > pixel_threshold:
                log_message = log_and_print(f"Warning: Canvas size ({output_width}x{output_height}, {canvas_pixels/1e6:.1f}M pixels) exceeds threshold ({pixel_threshold/1e6:.1f}M pixels) for multi-band blending.\n", log_message)
                log_message = log_and_print("Switching to 'Linear' blending for this step to conserve memory.\n", log_message)
                effective_blend_method = "linear"

            # Create output canvas - 4 Channels (BGRA)
            output_img = np.zeros((output_height, output_width, 4), dtype=np.uint8)

            # --- Calculate final transformation matrix for warping ---
            # This incorporates the translation onto the canvas
            final_warp_matrix_translated = None
            if is_affine:
                # We need the 2x3 matrix: (H_translation @ H_affine_3x3)[:2,:]
                final_warp_matrix_translated = (H_translation @ H_matrix_3x3_for_canvas)[:2, :]
            else:
                # We need the 3x3 matrix: H_translation @ H_homography_3x3
                final_warp_matrix_translated = H_translation @ H_matrix_3x3_for_canvas # H_matrix_3x3 holds the homography here

            # --- Warp img1 onto the canvas ---
            try:
                # Use borderValue=(0,0,0,0) for transparent borders
                if is_affine:
                    log_message = log_and_print("Warping image 1 using warpAffine (BGRA)...\n", log_message)
                    warped_img1_u8 = cv2.warpAffine(img1_u8, final_warp_matrix_translated, (output_width, output_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
                else:
                    log_message = log_and_print("Warping image 1 using warpPerspective (BGRA)...\n", log_message)
                    warped_img1_u8 = cv2.warpPerspective(img1_u8, final_warp_matrix_translated, (output_width, output_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
            except cv2.error as e_warp:
                 warp_type = 'Affine' if is_affine else 'Perspective'
                 log_message = log_and_print(f"Error during warping ({warp_type}): {e_warp}\n", log_message)
                 # Clean up before returning
                 if 'H_matrix_3x3_for_canvas' in locals():
                     del H_matrix_3x3_for_canvas
                 # final_warp_M was the matrix before translation
                 if 'final_warp_matrix_translated' in locals():
                     del final_warp_matrix_translated
                 if 'H_translation' in locals():
                     del H_translation
                 if 'img1_u8' in locals():
                     del img1_u8
                 if 'output_img' in locals():
                     del output_img
                 gc.collect()
                 return None, log_message

            # --- Clean up matrices and source image ---
            del H_matrix_3x3_for_canvas, H_translation, final_warp_matrix_translated, img1_u8
            # Note: final_warp_M (the untranslated matrix) is no longer needed
            if 'final_warp_M' in locals():
                del final_warp_M
            gc.collect()

            # Place img2 onto the canvas
            y_start, x_start = translation_dist[1], translation_dist[0]
            y_end, x_end = y_start + h2, x_start + w2

            # Define slicing for img2 read and canvas write, handling out-of-bounds placement
            img2_y_start, img2_x_start = 0, 0
            img2_y_end, img2_x_end = h2, w2
            canvas_y_start, canvas_x_start = y_start, x_start
            canvas_y_end, canvas_x_end = y_end, x_end
            # Clip coordinates
            if canvas_y_start < 0:
                img2_y_start = -canvas_y_start; canvas_y_start = 0
            if canvas_x_start < 0:
                img2_x_start = -canvas_x_start; canvas_x_start = 0
            if canvas_y_end > output_height:
                img2_y_end = h2 - (canvas_y_end - output_height); canvas_y_end = output_height
            if canvas_x_end > output_width:
                img2_x_end = w2 - (canvas_x_end - output_width); canvas_x_end = output_width

            # Check if the calculated slices are valid
            slice_h_canvas = canvas_y_end - canvas_y_start
            slice_w_canvas = canvas_x_end - canvas_x_start
            slice_h_img2 = img2_y_end - img2_y_start
            slice_w_img2 = img2_x_end - img2_x_start

            mask_img2 = np.zeros(output_img.shape[:2], dtype=np.uint8) # Mask for img2 placement
            img2_part = None
            if slice_h_canvas > 0 and slice_w_canvas > 0 and slice_h_canvas == slice_h_img2 and slice_w_canvas == slice_w_img2:
                img2_part = img2_u8[img2_y_start:img2_y_end, img2_x_start:img2_x_end]
                output_img[canvas_y_start:canvas_y_end, canvas_x_start:canvas_x_end] = img2_part
                # Mask based on Alpha channel of img2
                mask_img2[canvas_y_start:canvas_y_end, canvas_x_start:canvas_x_end] = (img2_part[:, :, 3] > 0).astype(np.uint8) * 255
            else:
                log_message = log_and_print("Warning: Could not place img2 correctly onto the canvas.\n", log_message)
            del img2_u8
            img2_u8 = None # Input img2 no longer needed
            gc.collect()

            # --- Create Masks for Blending ---
            # Create mask for the warped image 1 using Alpha Channel
            if warped_img1_u8 is not None:
                 # Check alpha channel (index 3)
                 mask_warped = (warped_img1_u8[:, :, 3] > 0).astype(np.uint8) * 255
                 # Erode the warped mask slightly to remove semi-transparent edge artifacts
                 # This removes the 1-pixel border caused by linear interpolation blending with transparency.
                 erosion_kernel = np.ones((3, 3), np.uint8)
                 mask_warped = cv2.erode(mask_warped, erosion_kernel, iterations=3)
            else:
                 mask_warped = np.zeros(output_img.shape[:2], dtype=np.uint8) # Empty mask if warp failed

            # Find overlapping region mask (uint8 0 or 255)
            overlap_mask = cv2.bitwise_and(mask_warped, mask_img2)
            has_overlap = np.sum(overlap_mask > 0) > 0 # Check if any pixel > 0
            log_message = log_and_print(f"Overlap detected: {has_overlap}\n", log_message)

            # --- Gain Compensation ---
            gain = 1.0
            gain_applied_warped_img1_u8 = warped_img1_u8 # Initialize with original warped image

            if enable_gain_compensation and has_overlap and warped_img1_u8 is not None: # Need warped image for gain comp
                log_message = log_and_print("Gain Compensation Enabled. Calculating gain...\n", log_message)
                try:
                    # --- Gain Calculation (Slice Logic for BGRA) ---
                    # Convert BGRA to GRAY for calculation
                    gray_warped_for_gain = cv2.cvtColor(warped_img1_u8, cv2.COLOR_BGRA2GRAY)
                    img2_gray = np.zeros_like(gray_warped_for_gain)
                    # Use the exact slice coordinates derived from placement step
                    if slice_h_canvas > 0 and slice_w_canvas > 0:
                        if 0 <= canvas_y_start < canvas_y_end <= output_height and \
                           0 <= canvas_x_start < canvas_x_end <= output_width:
                            # Slice from output_img (which is BGRA)
                            img_to_convert = output_img[canvas_y_start:canvas_y_end, canvas_x_start:canvas_x_end]
                            
                            if img_to_convert.size > 0:
                                # Convert Slice from BGRA to GRAY
                                img2_part_gray = cv2.cvtColor(img_to_convert, cv2.COLOR_BGRA2GRAY)
                                img2_gray[canvas_y_start:canvas_y_end, canvas_x_start:canvas_x_end] = img2_part_gray
                                del img2_part_gray
                            else: 
                                log_message = log_and_print("Warning: Empty slice for gain calculation img2_gray.\n", log_message)
                        else: 
                            log_message = log_and_print("Warning: Invalid slice indices for gain calculation img2_gray.\n", log_message)

                    overlap_mask_gain = overlap_mask # Use the already computed overlap mask
                    # Ensure masks are single channel before bitwise_and
                    
                    gray_warped_roi = cv2.bitwise_and(gray_warped_for_gain, gray_warped_for_gain, mask=overlap_mask_gain)
                    img2_roi = cv2.bitwise_and(img2_gray, img2_gray, mask=overlap_mask_gain)
                    del gray_warped_for_gain, img2_gray

                    overlap_pixel_count = np.sum(overlap_mask_gain > 0)
                    if overlap_pixel_count > 0:
                        # Ensure ROIs are valid before calculating sum
                        mean1 = np.sum(gray_warped_roi[overlap_mask_gain > 0]) / overlap_pixel_count if gray_warped_roi is not None else 0
                        mean2 = np.sum(img2_roi[overlap_mask_gain > 0]) / overlap_pixel_count if img2_roi is not None else 0

                        if mean1 > 1e-5 and mean2 > 1e-5:
                            gain = mean2 / mean1
                            log_message = log_and_print(f"Calculated Gain: {gain:.2f}\n", log_message)
                            gain = np.clip(gain, 0.5, 2.0) # Clamp gain
                            log_message = log_and_print(f"Clamped Gain: {gain:.2f}\n", log_message)
                        else:
                            gain = 1.0
                            log_message = log_and_print("Gain compensation skipped (means close to zero or invalid ROI).\n", log_message)
                    else:
                        gain = 1.0
                        log_message = log_and_print("Gain compensation skipped (no overlap pixels).\n", log_message)
                    del gray_warped_roi, img2_roi
                    gc.collect()
                    # --- End Gain Calculation ---

                    # Apply gain ONLY if calculated and different from 1.0
                    if abs(gain - 1.0) > 1e-5: # Check float difference
                        gain_applied_float = warped_img1_u8.astype(np.float32)
                        # Apply gain to RGB channels (0,1,2), Leave Alpha (3) untouched
                        gain_applied_float[:, :, :3] *= gain
                        
                        # *** Create new array for gain applied result ***
                        temp_gain_applied = gain_applied_float.clip(0, 255).astype(np.uint8)
                        # If gain_applied_warped_img1_u8 wasn't the original, delete it before reassigning
                        if gain_applied_warped_img1_u8 is not warped_img1_u8:
                            del gain_applied_warped_img1_u8
                        gain_applied_warped_img1_u8 = temp_gain_applied # Assign the new gain-applied image
                        del gain_applied_float, temp_gain_applied
                        gc.collect()
                        log_message = log_and_print(f"Gain applied to warped image (RGB channels).\n", log_message)
                    else:
                        log_message = log_and_print("Gain is ~1.0, no gain applied.\n", log_message)

                except Exception as e_gain_calc:
                    gain = 1.0
                    log_message = log_and_print(f"Warning: Error during gain calculation ({e_gain_calc}). Setting gain=1.0.\n", log_message)
                    # Ensure gain_applied remains the original warped image on error
                    if gain_applied_warped_img1_u8 is not warped_img1_u8:
                        del gain_applied_warped_img1_u8 # Delete potentially modified one
                        gc.collect()
                    gain_applied_warped_img1_u8 = warped_img1_u8 # Reset to original
                    # Clean up potential partial variables
                    if 'gray_warped_for_gain' in locals(): del gray_warped_for_gain
                    if 'img2_gray' in locals(): del img2_gray
                    if 'gray_warped_roi' in locals(): del gray_warped_roi
                    if 'img2_roi' in locals(): del img2_roi
                    if 'gain_applied_float' in locals(): del gain_applied_float
                    gc.collect()
            elif warped_img1_u8 is None:
                 log_message = log_and_print("Skipping Gain Compensation as warped image is None.\n", log_message)

            # Ensure gain_applied_warped_img1_u8 holds the image to be used for blending
            # (either original warped or gain-compensated version)

            # --- Blending Choice ---
            # Blend using the potentially gain-compensated image: gain_applied_warped_img1_u8
            if effective_blend_method == "multi-band" and has_overlap and gain_applied_warped_img1_u8 is not None:
                log_message = log_and_print(f"Applying Multi-band blending (Levels={num_blend_levels})...\n", log_message)
                try:
                    # --- Generate Blend Mask using Distance Transform ---
                    log_message = log_and_print("Generating multi-band mask using distance transform...\n", log_message)

                    # Distance Transform needs single channel mask
                    # masks are already single channel uint8
                    
                    dist1 = cv2.distanceTransform(mask_warped, cv2.DIST_L2, 5)
                    dist2 = cv2.distanceTransform(mask_img2, cv2.DIST_L2, 5)

                    # Create float32 weight mask
                    weight1_norm = np.zeros(output_img.shape[:2], dtype=np.float32)
                    
                    # Identify non-overlapping regions (ensure using single channel masks)
                    non_overlap_mask1 = cv2.bitwise_and(mask_warped, cv2.bitwise_not(overlap_mask))
                    non_overlap_mask2 = cv2.bitwise_and(mask_img2, cv2.bitwise_not(overlap_mask))

                    # Assign weights: 1.0 where only img1 exists, 0.0 where only img2 exists
                    weight1_norm[non_overlap_mask1 > 0] = 1.0
                    weight1_norm[non_overlap_mask2 > 0] = 0.0 # Implicitly 0 initially, but good to be explicit
                    
                    # Calculate weights in the overlap region based on relative distance
                    # Weight for img1 = dist1 / (dist1 + dist2)
                    overlap_indices = np.where(overlap_mask > 0)
                    num_overlap_pixels = len(overlap_indices[0])
                    if num_overlap_pixels > 0:
                        d1_overlap = dist1[overlap_indices]
                        d2_overlap = dist2[overlap_indices]
                        total_dist = d1_overlap + d2_overlap
                        # Avoid division by zero where total_dist is very small (deep inside both masks)
                        # If total_dist is near zero, assign weight based on which original mask was stronger?
                        # Using dist1 / (total_dist + epsilon) is simpler and generally works.
                        weights_overlap = d1_overlap / (total_dist + 1e-7) # Epsilon for stability
                        weight1_norm[overlap_indices] = np.clip(weights_overlap, 0.0, 1.0)
                        log_message = log_and_print(f"Calculated distance transform weights for {num_overlap_pixels} overlap pixels.\n", log_message)
                    else:
                        log_message = log_and_print("Warning: No overlap pixels found for distance transform weight calculation.\n", log_message)
                        
                    # Create boolean masks for later restoration steps
                    mask_warped_binary = (mask_warped > 0)
                    mask_img2_binary = (mask_img2 > 0)
                    overlap_mask_binary = (overlap_mask > 0)
                    
                    # Clean up intermediate arrays from distance transform step
                    del dist1, dist2, non_overlap_mask1, non_overlap_mask2, overlap_indices
                    if 'd1_overlap' in locals():
                        del d1_overlap
                    if 'd2_overlap' in locals():
                        del d2_overlap
                    if 'total_dist' in locals():
                        del total_dist
                    if 'weights_overlap' in locals():
                        del weights_overlap
                    gc.collect()
                    
                    # Apply Smoothing based on blend_smooth_ksize
                    blend_mask_float = weight1_norm # Start with the precise distance-based mask
                    if blend_smooth_ksize > 0 and blend_smooth_ksize % 2 == 1:
                        log_message = log_and_print(f"Smoothing multi-band blend mask with GaussianBlur ksize=({blend_smooth_ksize},{blend_smooth_ksize})...\n", log_message)
                        try:
                            # Need the boolean masks calculated above
                            
                            # Strict non-overlap areas (boolean arrays)
                            strict_non_overlap_mask1 = np.logical_and(mask_warped_binary, np.logical_not(overlap_mask_binary))
                            strict_non_overlap_mask2 = np.logical_and(mask_img2_binary, np.logical_not(overlap_mask_binary))


                            # Blur the original distance-based mask
                            weight1_norm_blurred = cv2.GaussianBlur(weight1_norm, (blend_smooth_ksize, blend_smooth_ksize), 0)

                            # Clip the blurred mask to [0, 1]
                            blend_mask_float_blurred = np.clip(weight1_norm_blurred, 0.0, 1.0)

                            # Assign the potentially blurred values first
                            blend_mask_float = blend_mask_float_blurred

                            # Force 1.0 where only img1 should be
                            blend_mask_float[strict_non_overlap_mask1] = 1.0
                            # Force 0.0 where only img2 should be
                            blend_mask_float[strict_non_overlap_mask2] = 0.0

                            log_message = log_and_print("Multi-band mask smoothed and edges restored.\n", log_message)

                        except Exception as e_blur_other:
                            log_message = log_and_print(f"Warning: Error during multi-band mask blur/restore ({e_blur_other}). Using original distance-based mask.\n", log_message)
                            blend_mask_float = weight1_norm # Fallback
                        finally:
                            # Clean up intermediate variables created in this block
                            if 'strict_non_overlap_mask1' in locals(): del strict_non_overlap_mask1
                            if 'strict_non_overlap_mask2' in locals(): del strict_non_overlap_mask2
                            if 'weight1_norm_blurred' in locals(): del weight1_norm_blurred
                            if 'blend_mask_float_blurred' in locals(): del blend_mask_float_blurred
                            gc.collect()
                    else:
                        log_message = log_and_print("Skipping multi-band mask smoothing (ksize not positive odd integer).\n", log_message)
                        # blend_mask_float is already weight1_norm (the precise one)
                    # --- End Smoothing ---

                    # --- Prepare for Blending ---
                    img1_for_blend = gain_applied_warped_img1_u8.astype(np.float32)
                    # Store the state of output_img BEFORE multi-band blending
                    output_img_before_mb_float = output_img.astype(np.float32)
                    img2_for_blend = output_img_before_mb_float # Use the float version

                    # --- Call Multi-Band Blending ---
                    # Note: Ensure multi_band_blending supports 4-channel images.
                    blended_result_uint8 = multi_band_blending(
                        img1_for_blend,
                        img2_for_blend,
                        blend_mask_float, # The prepared mask
                        num_levels=num_blend_levels
                    )

                    # --- Restore Non-Overlap Regions ---
                    log_message = log_and_print("Restoring non-overlap regions after multi-band blending...\n", log_message)

                    # Re-identify strict non-overlap boolean masks (using the ones calculated earlier)
                    strict_non_overlap_mask1 = np.logical_and(mask_warped_binary, np.logical_not(overlap_mask_binary))
                    strict_non_overlap_mask2 = np.logical_and(mask_img2_binary, np.logical_not(overlap_mask_binary))

                    # Convert blended result to float for modification
                    output_img_float = blended_result_uint8.astype(np.float32)

                    # Copy original pixels back into the non-overlap regions
                    # For img1's non-overlap region, use the (potentially gain compensated) warped img1
                    output_img_float[strict_non_overlap_mask1] = img1_for_blend[strict_non_overlap_mask1]

                    # For img2's non-overlap region, use the pixels from *before* blending
                    output_img_float[strict_non_overlap_mask2] = output_img_before_mb_float[strict_non_overlap_mask2]

                    # Convert back to uint8 for the final result for this step
                    output_img = np.clip(output_img_float, 0, 255).astype(np.uint8)
                    log_message = log_and_print("Non-overlap regions restored.\n", log_message)

                    # Apply final exterior mask (keep pure transparency clean)
                    combined_mask_binary = np.logical_or(mask_warped_binary, mask_img2_binary)
                    output_img[~combined_mask_binary] = 0 # Apply the sharp combined mask
                    log_message = log_and_print("Applied final exterior mask.\n", log_message)

                    # Cleanup
                    del img1_for_blend, img2_for_blend, output_img_before_mb_float, blend_mask_float
                    del blended_result_uint8, output_img_float
                    del mask_warped_binary, mask_img2_binary, overlap_mask_binary
                    del strict_non_overlap_mask1, strict_non_overlap_mask2
                    if 'combined_mask_binary' in locals():
                        del combined_mask_binary
                    if 'weight1_norm' in locals():
                        del weight1_norm
                    gc.collect()
                    log_message = log_and_print(f"Multi-band blending with restoration successful.\n", log_message)

                except Exception as e_blend:
                    log_message = log_and_print(f"Error during multi-band blending/restoration: {e_blend}. Falling back to simple overlay.\n{traceback.format_exc()}\n", log_message)
                    # Fallback: Use copyTo with 4-channel support (mask needs to be 1 channel or 4 channels)
                    # mask_warped is single channel uint8, works with 4-channel image in copyTo
                    output_img = cv2.copyTo(gain_applied_warped_img1_u8, mask_warped, output_img)

                    # Ensure cleanup if error happened mid-process
                    if 'img1_for_blend' in locals():
                        del img1_for_blend
                    if 'img2_for_blend' in locals():
                        del img2_for_blend
                    if 'output_img_before_mb_float' in locals():
                        del output_img_before_mb_float
                    if 'blend_mask_float' in locals():
                        del blend_mask_float
                    if 'blended_result_uint8' in locals():
                        del blended_result_uint8
                    if 'mask_warped_binary' in locals():
                        del mask_warped_binary # Clean up boolean masks too
                    if 'mask_img2_binary' in locals():
                        del mask_img2_binary
                    if 'overlap_mask_binary' in locals():
                        del overlap_mask_binary
                    gc.collect()

                    
            # Linear Blending
            elif effective_blend_method == "linear" and has_overlap and gain_applied_warped_img1_u8 is not None:
                log_message = log_and_print("Applying Linear blending...\n", log_message)
                # Ensure overlap_mask is single channel for findContours
                contours, _ = cv2.findContours(overlap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    log_message = log_and_print("Warning: No contours in overlap. Using simple overlay.\n", log_message)
                    output_img = cv2.copyTo(gain_applied_warped_img1_u8, mask_warped, output_img)
                else:
                    main_contour = max(contours, key=cv2.contourArea)
                    x_overlap, y_overlap, w_overlap, h_overlap = cv2.boundingRect(main_contour)
                    # Clip bounding box to canvas dimensions
                    x_overlap = max(0, x_overlap); y_overlap = max(0, y_overlap)
                    w_overlap = min(w_overlap, output_width - x_overlap); h_overlap = min(h_overlap, output_height - y_overlap)

                    if w_overlap <= 0 or h_overlap <= 0:
                        log_message = log_and_print("Warning: Invalid overlap bounding box after clipping. Using simple overlay.\n", log_message)
                        output_img = cv2.copyTo(gain_applied_warped_img1_u8, mask_warped, output_img)
                    else:
                        # Create weight maps (float32)
                        weight1 = np.zeros(output_img.shape[:2], dtype=np.float32)
                        weight2 = np.zeros(output_img.shape[:2], dtype=np.float32)
                        blend_axis = 0 if w_overlap >= h_overlap else 1
                        overlap_region_mask = overlap_mask[y_overlap : y_overlap + h_overlap, x_overlap : x_overlap + w_overlap]

                        # Generate gradient for the overlap box
                        gradient = None
                        if blend_axis == 0: # Horizontal blend
                            gradient = np.tile(np.linspace(1.0, 0.0, w_overlap, dtype=np.float32), (h_overlap, 1))
                        else: # Vertical blend
                            gradient = np.tile(np.linspace(1.0, 0.0, h_overlap, dtype=np.float32).reshape(-1, 1), (1, w_overlap))

                        weight1_region = gradient
                        weight2_region = 1.0 - gradient

                        # Apply weights only where the overlap mask is valid within the bounding box
                        valid_overlap = overlap_region_mask > 0
                        weight1[y_overlap : y_overlap + h_overlap, x_overlap : x_overlap + w_overlap][valid_overlap] = weight1_region[valid_overlap]
                        weight2[y_overlap : y_overlap + h_overlap, x_overlap : x_overlap + w_overlap][valid_overlap] = weight2_region[valid_overlap]
                        del weight1_region, weight2_region, gradient, valid_overlap, overlap_region_mask
                        gc.collect()

                        # Assign weights for non-overlapping regions (ensure masks are single channel)
                        non_overlap_mask1 = cv2.bitwise_and(mask_warped, cv2.bitwise_not(overlap_mask))
                        weight1[non_overlap_mask1 > 0] = 1.0
                        non_overlap_mask2 = cv2.bitwise_and(mask_img2, cv2.bitwise_not(overlap_mask))
                        weight2[non_overlap_mask2 > 0] = 1.0 
                        
                        # Normalize weights before potential smoothing
                        total_weight = weight1 + weight2 + 1e-6 # Add epsilon
                        weight1_norm = weight1 / total_weight
                        weight2_norm = weight2 / total_weight
                        del weight1, weight2, total_weight
                        gc.collect()
                        
                        # Apply Smoothing based on blend_smooth_ksize
                        if blend_smooth_ksize > 0 and blend_smooth_ksize % 2 == 1:
                            log_message = log_and_print(f"Smoothing linear blend weights with GaussianBlur ksize=({blend_smooth_ksize},{blend_smooth_ksize})...\n", log_message)
                            try:
                                # Identify the actual blending area (where both weights contribute meaningfully and overlap exists)
                                overlap_area_mask_bool = (weight1_norm > 1e-6) & (weight2_norm > 1e-6) & (overlap_mask > 0)

                                smoothed_w1 = cv2.GaussianBlur(weight1_norm, (blend_smooth_ksize, blend_smooth_ksize), 0)
                                smoothed_w2 = cv2.GaussianBlur(weight2_norm, (blend_smooth_ksize, blend_smooth_ksize), 0)

                                # Renormalize smoothed weights ONLY in the overlap area
                                total_smoothed_weight = smoothed_w1 + smoothed_w2 + 1e-6
                                # Use temporary arrays to avoid modifying originals during calculation if needed
                                temp_w1 = weight1_norm.copy() # Work on copies
                                temp_w2 = weight2_norm.copy()
                                temp_w1[overlap_area_mask_bool] = (smoothed_w1 / total_smoothed_weight)[overlap_area_mask_bool]
                                temp_w2[overlap_area_mask_bool] = (smoothed_w2 / total_smoothed_weight)[overlap_area_mask_bool]

                                # Restore strict 1.0 / 0.0 weights in non-overlap areas
                                temp_w1[ non_overlap_mask1 > 0 ] = 1.0
                                temp_w1[ non_overlap_mask2 > 0 ] = 0.0
                                temp_w2[ non_overlap_mask1 > 0 ] = 0.0
                                temp_w2[ non_overlap_mask2 > 0 ] = 1.0

                                # Assign back to the working variables
                                weight1_norm = temp_w1
                                weight2_norm = temp_w2

                                del smoothed_w1, smoothed_w2, total_smoothed_weight, overlap_area_mask_bool, temp_w1, temp_w2
                                gc.collect()
                                log_message = log_and_print("Linear weights smoothed and renormalized.\n", log_message)
                            except Exception as e_blur_other:
                                log_message = log_and_print(f"Warning: Error during linear weight smoothing ({e_blur_other}). Using original weights.\n", log_message)
                            finally:
                                # Ensure cleanup of temp vars in this block
                                if 'smoothed_w1' in locals():
                                    del smoothed_w1
                                if 'smoothed_w2' in locals():
                                    del smoothed_w2
                                if 'total_smoothed_weight' in locals():
                                    del total_smoothed_weight
                                if 'overlap_area_mask_bool' in locals():
                                    del overlap_area_mask_bool
                                if 'temp_w1' in locals():
                                    del temp_w1
                                if 'temp_w2' in locals():
                                    del temp_w2
                                gc.collect()
                        else:
                            log_message = log_and_print("Skipping linear weight smoothing (ksize not positive odd integer).\n", log_message)
                        # --- End Smoothing ---
                        
                        # Blend using potentially smoothed and renormalized weights
                        # Identify regions: where img1 only, img2 only, and blend region
                        non_overlap_mask1_bool = (non_overlap_mask1 > 0)
                        non_overlap_mask2_bool = (non_overlap_mask2 > 0)
                        blend_mask_bool = np.logical_not(np.logical_or(non_overlap_mask1_bool, non_overlap_mask2_bool)) & (overlap_mask > 0)
                        
                        # Copy non-overlapping part of image 1 directly where its weight is 1
                        output_img[non_overlap_mask1_bool] = gain_applied_warped_img1_u8[non_overlap_mask1_bool]
                        
                        # Non-overlapping part of image 2 is already in output_img from the initial placement

                        # Blend the overlapping/transition areas
                        blend_indices = np.where(blend_mask_bool)
                        num_blend_pixels = len(blend_indices[0])

                        if num_blend_pixels > 0:
                            log_message = log_and_print(f"Blending {num_blend_pixels} pixels linearly...\n", log_message)
                            try:
                                # Ensure images are float32 for blending calculation
                                img1_blend_float = gain_applied_warped_img1_u8[blend_indices].astype(np.float32)
                                img2_blend_float = output_img[blend_indices].astype(np.float32) # Pixels already placed from img2

                                # Get weights for the blend region and broadcast for element-wise multiplication
                                w1_blend_1d = weight1_norm[blend_indices]
                                w2_blend_1d = weight2_norm[blend_indices]
                                # Add new axis for broadcasting: (N,) -> (N, 1) to multiply with (N, 3) pixel data
                                w1_blend_broadcast = w1_blend_1d[:, np.newaxis]
                                w2_blend_broadcast = w2_blend_1d[:, np.newaxis]

                                # Perform the weighted sum
                                blended_float = w1_blend_broadcast * img1_blend_float + w2_blend_broadcast * img2_blend_float
                                blended_uint8 = blended_float.clip(0, 255).astype(np.uint8)

                                # Place the blended result back into the output image
                                output_img[blend_indices] = blended_uint8

                                del img1_blend_float, img2_blend_float, w1_blend_1d, w2_blend_1d
                                del w1_blend_broadcast, w2_blend_broadcast, blended_float, blended_uint8
                                gc.collect()
                                log_message = log_and_print("Linear blending successful.\n", log_message)

                            except Exception as e_blend_lin:
                                log_message = log_and_print(f"Warning: Error during float blending ({e_blend_lin}). Using simple overlay for blend region.\n", log_message)
                                blend_mask_uint8 = blend_mask_bool.astype(np.uint8) * 255
                                if np.any(blend_mask_uint8):
                                    output_img = cv2.copyTo(gain_applied_warped_img1_u8, blend_mask_uint8, output_img)
                                del blend_mask_uint8
                                gc.collect()
                        else:
                            log_message = log_and_print("Note: Linear blend mask was empty, skipping float blend step.\n", log_message)

                        # Clean up linear blending specific variables
                        del weight1_norm, weight2_norm, blend_mask_bool
                        del non_overlap_mask1, non_overlap_mask2, non_overlap_mask1_bool, non_overlap_mask2_bool
                        if 'blend_indices' in locals():
                            del blend_indices
                        gc.collect()

                # Clean up contour variables regardless of path taken inside linear blend
                if 'contours' in locals():
                    del contours
                if 'main_contour' in locals():
                    del main_contour
                gc.collect()

            # Simple overlay if no blending applied or specified OR if warped image was None
            elif not has_overlap or effective_blend_method not in ["linear", "multi-band"] or gain_applied_warped_img1_u8 is None:
                if gain_applied_warped_img1_u8 is None:
                    log_message = log_and_print("Warped image was None. Performing simple overlay (only showing img2).\n", log_message)
                    # In this case, output_img already contains img2 where it should be, and black elsewhere.
                    # No copyTo needed, as there's nothing to copy from.
                elif not has_overlap:
                    log_message = log_and_print("No overlap. Performing simple overlay.\n", log_message)
                else:
                    log_message = log_and_print(f"Blending method '{effective_blend_method}' or overlap condition not met. Performing simple overlay.\n", log_message)

                if gain_applied_warped_img1_u8 is not None: # Only copy if we have something to copy
                    output_img = cv2.copyTo(gain_applied_warped_img1_u8, mask_warped, output_img)

            # --- Final Result Assignment ---
            final_output_img = output_img # Assign the final blended/overlaid image

            end_time_pairwise = time.time()
            log_message = log_and_print(f"Pairwise stitching finished. Time: {end_time_pairwise - start_time_pairwise:.2f}s\n", log_message)

        else: # Not enough good matches
            log_message = log_and_print(f"Error: Not enough good matches ({len(good_matches)} < {MIN_MATCH_COUNT}).\n", log_message)
            # Minimal cleanup needed here, mostly handled in finally block
            if 'kp1' in locals():
                del kp1
            if 'kp2' in locals():
                del kp2
            if 'good_matches' in locals():
                del good_matches

    except Exception as e:
        log_message = log_and_print(f"Error during pairwise stitching: {e}\n{traceback.format_exc()}\n", log_message)
        final_output_img = None # Ensure None is returned on error

    finally:
        # --- Comprehensive Cleanup ---
        # Delete variables in roughly reverse order of creation / dependency
        # Blend-specific intermediates
        if 'img1_for_blend' in locals():
            del img1_for_blend
        if 'img2_for_blend' in locals():
            del img2_for_blend
        if 'output_img_before_mb_float' in locals():
            del output_img_before_mb_float
        if 'blend_mask_float' in locals():
            del blend_mask_float
        if 'weight1_norm' in locals():
            del weight1_norm # From mask gen (MB or Linear)
        if 'weight2_norm' in locals():
            del weight2_norm # From Linear mask gen
        # ... other linear/MB intermediate vars ...

        # Gain/Warp intermediates
        if 'gain_applied_warped_img1_u8' in locals() and gain_applied_warped_img1_u8 is not None:
            # Only delete if it's a separate copy from warped_img1_u8
            if 'warped_img1_u8' in locals() and warped_img1_u8 is not None and gain_applied_warped_img1_u8 is not warped_img1_u8:
                del gain_applied_warped_img1_u8
            # else it points to warped_img1_u8 or warped_img1_u8 is None/deleted already

        if 'warped_img1_u8' in locals() and warped_img1_u8 is not None:
            del warped_img1_u8
        if 'mask_warped' in locals():
            del mask_warped
        if 'mask_img2' in locals():
            del mask_img2
        if 'overlap_mask' in locals():
            del overlap_mask
        if 'img2_part' in locals():
            del img2_part # From placing img2

        if 'output_img' in locals() and output_img is not None and output_img is not final_output_img:
            # Delete intermediate output_img if it wasn't the final result (e.g., error occurred)
            del output_img

        # Transformation matrices and points
        if 'H_matrix_3x3_for_canvas' in locals():
            del H_matrix_3x3_for_canvas
        if 'final_warp_M' in locals():
            del final_warp_M
        if 'mask_trans' in locals():
            del mask_trans
        if 'src_pts' in locals():
            del src_pts
        if 'dst_pts' in locals():
            del dst_pts

        # Feature matching intermediates
        if 'kp1' in locals():
            del kp1
        if 'kp2' in locals():
            del kp2
        if 'des1' in locals():
            del des1
        if 'des2' in locals():
            del des2
        if 'good_matches' in locals():
            del good_matches
        if 'all_matches' in locals():
            del all_matches

        # Initial uint8 images
        if 'img1_u8' in locals():
            del img1_u8
        if 'img2_u8' in locals():
            del img2_u8

        gc.collect()

    return final_output_img, log_message

# --- Function for N-Image Stitching (Primarily for Image List Input) ---
def stitch_multiple_images(images, # List of NumPy images (BGR/BGRA, potentially pre-cropped)
                                    stitcher_mode_str="SCANS",
                                    registration_resol=0.6,
                                    seam_estimation_resol=0.1,
                                    compositing_resol=-1.0, # Use -1.0 for default/auto
                                    wave_correction=False,
                                    exposure_comp_type_str="GAIN_BLOCKS",
                                    enable_cropping=True, # This is for POST-stitch cropping
                                    strict_no_black_edges=False,
                                    # Pairwise/Fallback specific params
                                    transform_model_str="Homography",
                                    blend_method="multi-band",
                                    enable_gain_compensation=True,
                                    orb_nfeatures=2000,
                                    match_ratio_thresh=0.75,
                                    ransac_reproj_thresh=5.0,
                                    max_distance_coeff=0.5,
                                    max_blending_width=10000,
                                    max_blending_height=10000,
                                    blend_smooth_ksize=15,
                                    num_blend_levels=4
                                    ):
    """
    Stitches a list of images. Tries cv2.Stitcher first (unless 'DIRECT_PAIRWISE'),
    otherwise falls back to manual pairwise stitching.
    Returns ONE stitched image (RGB/RGBA) and log.
    Input images should be in BGR format (already potentially cropped by caller).
    Output is RGBA. The 'enable_cropping' param here refers to final black border cropping.
    """
    log = log_and_print(f"--- Starting Stitching Process for {len(images)} Provided Images ---\n", "")
    total_start_time = time.time()
    stitched_img_rgba = None # Initialize result

    if len(images) < 2:
        log = log_and_print("Error: Need at least two images to stitch.\n", log)
        return None, log

    # Check if any input image is None or empty after potential pre-cropping
    valid_images = []
    for i, img in enumerate(images):
        if img is None or img.size == 0:
            log = log_and_print(f"Warning: Input image at index {i} is invalid (None or empty). Skipping it.\n", log)
        else:
            valid_images.append(img)

    if len(valid_images) < 2:
        log = log_and_print(f"Error: Not enough valid images ({len(valid_images)}) left after checking. Cannot stitch.\n", log)
        del images, valid_images # Clean up
        gc.collect()
        return None, log

    images = valid_images # Use the filtered list
    log = log_and_print(f"Proceeding with {len(images)} valid images.\n", log)
    log = log_and_print(f"Selected Stitcher Mode: {stitcher_mode_str}\n", log)

    # Log the pairwise transform model choice, relevant if fallback or DIRECT_PAIRWISE
    if stitcher_mode_str == "DIRECT_PAIRWISE":
        log = log_and_print(f"Using Pairwise Transform Model: {transform_model_str}\n", log)
        log = log_and_print(f"Pairwise Params: RANSAC Thresh={ransac_reproj_thresh}, Max Dist Coeff={max_distance_coeff}\n", log)
    else:
        log = log_and_print(f"Pairwise Transform Model (for fallback): {transform_model_str}\n", log)
        log = log_and_print(f"Fallback Pairwise Params: RANSAC Thresh={ransac_reproj_thresh}, Max Dist Coeff={max_distance_coeff}\n", log)
    log = log_and_print(f"Post-Crop: Enable={enable_cropping}, Strict Edges={strict_no_black_edges}\n", log) # Log new param
    
    skip_cv2_stitcher = (stitcher_mode_str == "DIRECT_PAIRWISE")

    stitched_img_bgra = None
    stitcher_success = False

    # 1. Try using cv2.Stitcher (unless skipped)
    if not skip_cv2_stitcher:
        log = log_and_print("\nAttempting stitching with built-in cv2.Stitcher...\n", log)
        # ... [Existing cv2.Stitcher setup code omitted for brevity, logic remains same] ...
        # Note: cv2.Stitcher usually outputs BGR (3 channels), removing alpha. 
        # If transparency is critical, DIRECT_PAIRWISE is preferred.
        
        # Map string parameters to OpenCV constants for cv2.Stitcher modes
        stitcher_mode_map = {"PANORAMA": cv2.Stitcher_PANORAMA, "SCANS": cv2.Stitcher_SCANS}
        # Default to SCANS if invalid string for cv2.Stitcher mode itself
        cv2_stitcher_mode_enum = stitcher_mode_map.get(stitcher_mode_str, cv2.Stitcher_SCANS)
        log = log_and_print(f"Using OpenCV Stitcher Mode Enum: {cv2_stitcher_mode_enum} (from string: {stitcher_mode_str})\n", log)
        
        exposure_comp_map = {
            "NO": cv2.detail.ExposureCompensator_NO,
            "GAIN": cv2.detail.ExposureCompensator_GAIN,
            "GAIN_BLOCKS": cv2.detail.ExposureCompensator_GAIN_BLOCKS
        }
        exposure_comp_type = exposure_comp_map.get(exposure_comp_type_str, cv2.detail.ExposureCompensator_GAIN_BLOCKS)
        log = log_and_print(f"Using Exposure Compensation: {exposure_comp_type_str}\n", log)
        log = log_and_print(f"Wave Correction Enabled: {wave_correction}\n", log)

        stitcher = None # Initialize stitcher object variable
        try:
            stitcher = cv2.Stitcher.create(cv2_stitcher_mode_enum)
            if stitcher is None:
                raise RuntimeError("cv2.Stitcher.create returned None.")

            log = log_and_print(f"Setting Stitcher resolutions: Reg={registration_resol:.2f}, Seam={seam_estimation_resol:.2f}, Comp={compositing_resol:.2f}\n", log)
            try:
                if hasattr(stitcher, 'setRegistrationResol'):
                 stitcher.setRegistrationResol(float(registration_resol))
                if hasattr(stitcher, 'setSeamEstimationResol'):
                 stitcher.setSeamEstimationResol(float(seam_estimation_resol))
                if hasattr(stitcher, 'setCompositingResol'):
                 stitcher.setCompositingResol(float(compositing_resol))
            except Exception as e_res:
                log = log_and_print(f"Warning: Could not set stitcher resolutions: {e_res}\n", log)

            try:
                if hasattr(stitcher, 'setWaveCorrection'):
                    stitcher.setWaveCorrection(wave_correction)
            except Exception as e_wave:
                log = log_and_print(f"Warning: Could not set wave correction: {e_wave}\n", log)
                
            try:
                if hasattr(stitcher, 'setExposureCompensator'):
                    compensator = cv2.detail.ExposureCompensator_createDefault(exposure_comp_type)
                    stitcher.setExposureCompensator(compensator)
                    del compensator # Release compensator object reference
            except Exception as e_exp:
                log = log_and_print(f"Warning: Could not set exposure compensator: {e_exp}\n", log)

            # Note: cv2.Stitcher might strip alpha here.
            # Ensure all images are uint8 before passing to stitcher
            images_uint8 = []
            for img in images:
                if img.dtype != np.uint8:
                    images_uint8.append(img.clip(0, 255).astype(np.uint8))
                else:
                    images_uint8.append(img)

            status = cv2.Stitcher_ERR_NEED_MORE_IMGS # Initialize status to a known failure code
            stitched_img_raw = None
            
            try:
                log = log_and_print("Executing stitcher.stitch()...\n", log)
                status, stitched_img_raw = stitcher.stitch(images_uint8) # Input 'images' should be BGR uint8
                log = log_and_print(f"stitcher.stitch() returned status: {status}\n", log) # Log the status code

            except cv2.error as e_stitch:
                log = log_and_print(f"OpenCV Error occurred DURING stitcher.stitch() call: {e_stitch}\n", log)
                log = log_and_print(f"Traceback:\n{traceback.format_exc()}\n", log)
                log = log_and_print("Falling back to manual pairwise stitching method due to stitch() error.\n", log)
                status = -99 # Set status to a custom failure code to ensure fallback
                stitched_img_raw = None
            except Exception as e_stitch_other:
                log = log_and_print(f"Unexpected Error occurred DURING stitcher.stitch() call: {e_stitch_other}\n", log)
                log = log_and_print(f"Traceback:\n{traceback.format_exc()}\n", log)
                log = log_and_print("Falling back to manual pairwise stitching method due to unexpected stitch() error.\n", log)
                status = -100 # Set status to a custom failure code
                stitched_img_raw = None
            finally:
                del images_uint8
                gc.collect()

            if status == cv2.Stitcher_OK:
                log = log_and_print("cv2.Stitcher successful!\n", log)
                if stitched_img_raw is not None and stitched_img_raw.size > 0:
                    log = log_and_print(f"Stitcher output dimensions (raw): {stitched_img_raw.shape}\n", log)
                    # Apply FINAL black border cropping if enabled
                    cropped_result = crop_black_borders(stitched_img_raw, enable_cropping, strict_no_black_edges)
                    if cropped_result is not None and cropped_result.size > 0 :
                         stitched_img_bgra = cropped_result
                         log = log_and_print(f"Final dimensions after POST-stitch cropping: {stitched_img_bgra.shape}\n", log)
                    else:
                         stitched_img_bgra = stitched_img_raw
                         log = log_and_print("POST-stitch cropping failed or disabled, using raw stitcher output.\n", log)
                    stitcher_success = True
                    del stitched_img_raw
                    if 'cropped_result' in locals() and cropped_result is not stitched_img_bgra:
                        del cropped_result
                    gc.collect()
                else:
                    log = log_and_print("Error: cv2.Stitcher returned status OK but the image is empty.\n", log)
            else:
                error_codes = { getattr(cv2, k): k for k in dir(cv2) if k.startswith('Stitcher_ERR_') }
                error_codes[-99] = "ERR_STITCH_CV_ERROR"
                error_codes[-100] = "ERR_STITCH_EXCEPTION"
                # Check if fallback message was already logged by exceptions during stitch()
                if "Falling back to manual pairwise stitching method due to" not in log.splitlines()[-5:]:
                    log = log_and_print(f"cv2.Stitcher failed with status code: {status} ({error_codes.get(status, f'Unknown Error {status}')})\n", log)
                    log = log_and_print("Falling back to manual pairwise stitching method...\n", log)

        except AttributeError as e_attr:
            log = log_and_print(f"AttributeError during Stitcher setup ({e_attr}). Falling back.\n{traceback.format_exc()}\n", log)
        except RuntimeError as e_runtime:
            log = log_and_print(f"RuntimeError during Stitcher setup ({e_runtime}). Falling back.\n{traceback.format_exc()}\n", log)
        except cv2.error as e:
            log = log_and_print(f"OpenCV Error during Stitcher operation: {e}. Falling back.\n", log)
            if "OutOfMemoryError" in str(e) or "Insufficient memory" in str(e):
                 log = log_and_print(">>> Specific OutOfMemoryError detected. Reduce resolutions or use more RAM.\n", log)
            log = log_and_print(f"{traceback.format_exc()}\n", log)
        except Exception as e:
            log = log_and_print(f"Unexpected error during Stitcher: {e}. Falling back.\n{traceback.format_exc()}\n", log)
        finally:
            if stitcher is not None:
                # Attempt to release stitcher resources if possible (may not exist)
                try:
                    del stitcher
                except NameError:
                    pass
            gc.collect()
            
    # 2. Fallback or Direct Pairwise Stitching
    # Trigger if cv2.Stitcher was skipped OR if it failed
    if skip_cv2_stitcher or not stitcher_success:
        # Add clearer logging based on the reason
        if skip_cv2_stitcher:
            log = log_and_print(f"\n--- Starting Sequential Pairwise Stitching (Direct Mode, Transform: {transform_model_str}) ---\n", log)
        else:
            log = log_and_print(f"\n--- Starting Sequential Pairwise Stitching (Fallback, Transform: {transform_model_str}) ---\n", log)

        if len(images) >= 2:
            # Start with the first valid image. Ensure it's uint8.
            if images[0].dtype != np.uint8:
                current_stitched_image = images[0].clip(0, 255).astype(np.uint8)
            else:
                current_stitched_image = images[0].copy() # Copy to avoid modifying original list item

            sequential_stitch_success = True
            for i in range(1, len(images)):
                log = log_and_print(f"\nSequentially stitching image {i+1} of {len(images)} using pairwise method...\n", log)

                # Ensure next image is uint8
                if images[i].dtype != np.uint8:
                    next_image = images[i].clip(0, 255).astype(np.uint8)
                else:
                    next_image = images[i] # Can use directly if already uint8

                result, pairwise_log = stitch_pairwise_images(
                    current_stitched_image,       # BGR uint8
                    next_image,                   # BGR uint8
                    transform_model_str=transform_model_str,
                    blend_method=blend_method,
                    enable_gain_compensation=enable_gain_compensation,
                    orb_nfeatures=orb_nfeatures,
                    match_ratio_thresh=match_ratio_thresh,
                    ransac_reproj_thresh=ransac_reproj_thresh,
                    max_distance_coeff=max_distance_coeff,
                    max_blending_width=max_blending_width,
                    max_blending_height=max_blending_height,
                    blend_smooth_ksize=blend_smooth_ksize,
                    num_blend_levels=num_blend_levels
                )
                log += pairwise_log

                if result is None:
                    log = log_and_print(f"Error: Failed to stitch image {i+1} onto previous composite in the pairwise step. Aborting sequential process.\n", log) # Corrected index in log
                    sequential_stitch_success = False
                    if 'current_stitched_image' in locals() and current_stitched_image is not None:
                        del current_stitched_image # Clean up intermediate result
                    gc.collect()
                    break

                # Release the previous intermediate image before assigning the new one
                if 'current_stitched_image' in locals() and current_stitched_image is not None:
                    del current_stitched_image
                    gc.collect()
                current_stitched_image = result # Result is BGR uint8
                log = log_and_print(f"Intermediate stitched shape: {current_stitched_image.shape}\n", log)
                # Ensure next_image is cleaned up if it was a conversion
                if next_image is not images[i]:
                    del next_image
                    gc.collect()

            if sequential_stitch_success and current_stitched_image is not None:
                log = log_and_print("\nSequential pairwise stitching complete. Applying final cropping...\n", log)
                # Apply FINAL black border cropping if enabled
                cropped_fallback = crop_black_borders(current_stitched_image, enable_cropping, strict_no_black_edges)
                if cropped_fallback is not None and cropped_fallback.size > 0:
                    stitched_img_bgra = cropped_fallback
                    log = log_and_print(f"Final dimensions after POST-stitch cropping: {stitched_img_bgra.shape}\n", log)
                else:
                    stitched_img_bgra = current_stitched_image # Use uncropped if cropping failed
                    log = log_and_print("POST-stitch cropping failed or disabled, using uncropped manual result.\n", log)
                # Clean up the last intermediate/uncropped result if cropping was successful and created a new object
                if cropped_fallback is not current_stitched_image and current_stitched_image is not None:
                    del current_stitched_image
                if 'cropped_fallback' in locals() and cropped_fallback is not stitched_img_bgra:
                    del cropped_fallback
                gc.collect()
            else:
                log = log_and_print("Sequential pairwise stitching process could not produce a final result.\n", log)
                # Ensure cleanup if loop broke early or current_stitched_image was None/deleted
                if 'current_stitched_image' in locals() and current_stitched_image is not None:
                    del current_stitched_image
                    gc.collect()
        else: # Handle len(images) < 2 case (shouldn't happen due to initial check, but safety)
                 log = log_and_print("Error: Not enough images for pairwise stitching (internal check).\n", log)

    # Clean up the input image list now that it's processed
    del images
    if 'valid_images' in locals(): del valid_images # Should be same as images now
    gc.collect()

    # 3. Final Result Check and Return (Handling Alpha Channel)
    total_end_time = time.time()
    log = log_and_print(f"\nTotal processing time: {total_end_time - total_start_time:.2f} seconds.\n", log)

    if stitched_img_bgra is not None and stitched_img_bgra.size > 0:
        log = log_and_print("Stitching process finished for image list.", log)
        try:
            # Check channel count to preserve transparency
            if stitched_img_bgra.shape[2] == 4:
                # Convert BGRA to RGBA
                stitched_img_rgba = cv2.cvtColor(stitched_img_bgra, cv2.COLOR_BGRA2RGBA)
                log = log_and_print("Converted BGRA to RGBA (Transparency preserved).", log)
            else:
                # Convert BGR to RGB
                stitched_img_rgba = cv2.cvtColor(stitched_img_bgra, cv2.COLOR_BGR2RGB)
                log = log_and_print("Converted BGR to RGB.", log)
                
            del stitched_img_bgra # Release BGRA version
            gc.collect()
            return stitched_img_rgba, log
        except cv2.error as e_cvt:
            log = log_and_print(f"\nError converting final image: {e_cvt}. Returning None.\n", log)
            if 'stitched_img_bgr' in locals(): del stitched_img_bgra
            gc.collect()
            return None, log
    else:
        log = log_and_print("Error: Stitching failed. No final image generated.", log)
        if 'stitched_img_bgr' in locals() and stitched_img_bgra is not None:
            del stitched_img_bgra
            gc.collect()
        return None, log


# --- Video Frame Stitching ---
def stitch_video_frames(video_path,
                        crop_top_percent=0.0,
                        crop_bottom_percent=0.0,
                        enable_cropping=True, # This is for POST-stitch cropping
                        strict_no_black_edges=False,
                        # Pairwise specific params
                        transform_model_str="Homography",
                        blend_method="multi-band",
                        enable_gain_compensation=True,
                        orb_nfeatures=2000,
                        match_ratio_thresh=0.75,
                        ransac_reproj_thresh=5.0,
                        max_distance_coeff=0.5,
                        max_blending_width=10000,
                        max_blending_height=10000,
                        blend_smooth_ksize=15,
                        num_blend_levels=4,
                        # Video specific params
                        sample_interval_ms=3000,
                        max_composite_width=10000,
                        max_composite_height=10000,
                        progress=None):
    """
    Reads a video, samples frames, and stitches them sequentially.
    Returns a list of stitched images (RGBA/RGB) and a log.
    """
    log = log_and_print(f"--- Starting Incremental Video Stitching: {os.path.basename(video_path)} ---\n", "")
    log = log_and_print(f"Params: Interval={sample_interval_ms}ms, Transform={transform_model_str}, ORB={orb_nfeatures}, Ratio={match_ratio_thresh}\n", log)
    log = log_and_print(f"Params Cont'd: RANSAC Thresh={ransac_reproj_thresh}, Max Dist Coeff={max_distance_coeff}\n", log)
    log = log_and_print(f"Composite Limits: MaxW={max_composite_width}, MaxH={max_composite_height}\n", log)
    log = log_and_print(f"Pre-Crop: Top={crop_top_percent}%, Bottom={crop_bottom_percent}%\n", log)
    log = log_and_print(f"Post-Crop Black Borders: {enable_cropping}, Strict Edges: {strict_no_black_edges}\n", log)
    log = log_and_print(f"Blending: Method={blend_method}, GainComp={enable_gain_compensation}, SmoothKSize={blend_smooth_ksize}, MB Levels={num_blend_levels}\n", log)
    total_start_time = time.time()
    stitched_results_rgba = [] # Store final RGBA images

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log = log_and_print(f"Error: Could not open video file: {video_path}\n", log)
        return [], log

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or np.isnan(fps): # Handle invalid FPS reads
        fps = 30 # Default FPS
        log = log_and_print("Warning: Could not read valid FPS, defaulting to 30.\n", log)
    if frame_count_total <= 0: # Handle invalid frame count reads
        log = log_and_print("Warning: Could not read valid total frame count.\n", log)
        total_sampled_estimate = 0 # Cannot estimate progress accurately
    else:
        # Estimate total frames to be sampled, avoid division by zero if interval is 0
        frames_per_sample = max(1, int(round(fps * (sample_interval_ms / 1000.0)))) if sample_interval_ms > 0 else frame_count_total
        total_sampled_estimate = frame_count_total / frames_per_sample if frames_per_sample > 0 else 0


    frame_interval = max(1, int(round(fps * (sample_interval_ms / 1000.0))))
    log = log_and_print(f"Video Info: ~{fps:.2f} FPS, {frame_count_total} Frames, Sampling every {frame_interval} frames.\n", log)
    
    frame_num = 0
    processed_sampled_count = 0 # Counter for progress bar
    anchor_frame = None         # The starting frame of the current sequence (BGR, cropped)
    current_composite = None    # The stitched result being built (BGR, uint8)
    last_saved_composite = None # Keep track of the last saved image to avoid duplicates

    # Helper to save image to results list with correct color conversion
    def append_result(img_bgr_or_bgra):
        try:
            if img_bgr_or_bgra.shape[2] == 4:
                stitched_results_rgba.append(cv2.cvtColor(img_bgr_or_bgra, cv2.COLOR_BGRA2RGBA))
            else:
                stitched_results_rgba.append(cv2.cvtColor(img_bgr_or_bgra, cv2.COLOR_BGR2RGB))
            return True
        except Exception as e:
            print(f"Error converting result: {e}")
            return False

    while True:
        frame_bgr_raw = None # Initialize here for cleanup later
        try:
            if cap is None or not cap.isOpened():
                log = log_and_print("Error: Video capture became invalid during processing.\n", log)
                break
            ret, frame_bgr_raw = cap.read()
            if not ret:
                    log = log_and_print("\nEnd of video stream reached.\n", log)
                    break # End of video

                # --- Sampling Logic ---
            if frame_num % frame_interval == 0:
                if frame_bgr_raw is not None and frame_bgr_raw.size > 0:
                    processed_sampled_count += 1
                    frame_bgra = None # Initialize BGR frame variable
                
                    # Ensure BGRA for processing
                    if frame_bgr_raw.ndim == 2:
                        frame_bgra = cv2.cvtColor(frame_bgr_raw, cv2.COLOR_GRAY2BGRA)
                    elif frame_bgr_raw.ndim == 3 and frame_bgr_raw.shape[2] == 3:
                        # Add alpha channel (opaque)
                        frame_bgra = cv2.cvtColor(frame_bgr_raw, cv2.COLOR_BGR2BGRA)
                    elif frame_bgr_raw.ndim == 3 and frame_bgr_raw.shape[2] == 4:
                        frame_bgra = frame_bgr_raw
                    else:
                        log = log_and_print(f"Warning: Skipping frame {frame_num} due to unexpected shape {frame_bgr_raw.shape}\n", log)
                        if frame_bgr_raw is not None:
                            del frame_bgr_raw # Clean up original frame
                        gc.collect()
                        frame_num += 1
                        continue # Skip to next frame read
                    
                    # Release the raw frame once converted/checked (if a copy was made)
                    if frame_bgra is not frame_bgr_raw:
                        del frame_bgr_raw
                        frame_bgr_raw = None # Mark as deleted
                        gc.collect()

                    cropped_frame_bgra = crop_image_by_percent(frame_bgra, crop_top_percent, crop_bottom_percent)
                    del frame_bgra # Release the uncropped BGR version
                    gc.collect()

                    # Check if cropping failed or resulted in an empty image
                    if cropped_frame_bgra is None or cropped_frame_bgra.size == 0:
                        log = log_and_print(f"Warning: Skipping frame {frame_num} because percentage cropping failed or resulted in empty image.\n", log)
                        if cropped_frame_bgra is not None: del cropped_frame_bgra # Should be None, but safety check
                        gc.collect()
                        frame_num += 1
                        continue # Skip to next frame read
                    
                    # Now use 'cropped_frame_bgra' as the current frame for stitching
                    current_frame_for_stitch = cropped_frame_bgra # BGRA, uint8, potentially cropped
                    
                    if progress is not None and total_sampled_estimate > 0:
                        # Ensure progress doesn't exceed 1.0
                        progress_fraction = min(1.0, processed_sampled_count / total_sampled_estimate)
                        progress(progress_fraction, desc=f"Processing Sample {processed_sampled_count}/{int(total_sampled_estimate)}")
                    elif progress is not None:
                        # Fallback progress if estimate is bad
                         progress(processed_sampled_count / (processed_sampled_count + 10), desc=f"Processing Sample {processed_sampled_count}")
                         

                    log = log_and_print(f"\n--- Processing sampled frame index {frame_num} (Count: {processed_sampled_count}) ---\n", log)
                    log = log_and_print(f"Frame shape after potential pre-crop: {current_frame_for_stitch.shape}\n", log)

                    # --- Stitching Logic ---
                    if anchor_frame is None:
                        # Start a new sequence
                        anchor_frame = current_frame_for_stitch.copy() # Make a copy
                        current_composite = anchor_frame # Start composite is the anchor itself
                        log = log_and_print(f"Frame {frame_num}: Set as new anchor (Shape: {anchor_frame.shape}).\n", log)
                        # No need to stitch yet, just set the anchor

                    else:
                        # Try stitching the current composite with the new frame
                        log = log_and_print(f"Attempting stitch: Composite({current_composite.shape}) + Frame({current_frame_for_stitch.shape})\n", log)

                        stitch_result, stitch_log = stitch_pairwise_images(
                            current_composite,          # Previous result or anchor (uint8)
                            current_frame_for_stitch,   # New frame to add (uint8)
                            transform_model_str=transform_model_str,
                            blend_method=blend_method,
                            enable_gain_compensation=enable_gain_compensation,
                            orb_nfeatures=orb_nfeatures,
                            match_ratio_thresh=match_ratio_thresh,
                            ransac_reproj_thresh=ransac_reproj_thresh,
                            max_distance_coeff=max_distance_coeff,
                            max_blending_width=max_blending_width,
                            max_blending_height=max_blending_height,
                            blend_smooth_ksize=blend_smooth_ksize,
                            num_blend_levels=num_blend_levels
                        )
                        log += stitch_log

                        if stitch_result is not None and stitch_result.size > 0:
                            # --- Stitching SUCCEEDED ---
                            log = log_and_print(f"Success: Stitched frame {frame_num}. New composite shape: {stitch_result.shape}\n", log)
                            # Release old composite before assigning new one
                            del current_composite
                            gc.collect()
                            current_composite = stitch_result # Update the composite (stitch_result is BGR uint8)
                            # anchor_frame remains the same for this sequence

                            # Check Size Limits
                            h_curr, w_curr = current_composite.shape[:2]
                            size_limit_exceeded = False
                            # Check only if limit > 0
                            if max_composite_width > 0 and w_curr > max_composite_width:
                                log = log_and_print(f"ACTION: Composite width ({w_curr}) exceeded limit ({max_composite_width}).\n", log)
                                size_limit_exceeded = True
                            if max_composite_height > 0 and h_curr > max_composite_height:
                                log = log_and_print(f"ACTION: Composite height ({h_curr}) exceeded limit ({max_composite_height}).\n", log)
                                size_limit_exceeded = True
                                
                            if size_limit_exceeded:
                                log = log_and_print("Saving current composite and starting new sequence with NEXT frame.\n", log)

                                # Apply FINAL black border cropping if enabled
                                post_cropped_composite = crop_black_borders(current_composite, enable_cropping, strict_no_black_edges)
                                if post_cropped_composite is not None and post_cropped_composite.size > 0:
                                    # Avoid saving the exact same image twice in a row
                                    is_duplicate = False
                                    if last_saved_composite is not None:
                                        try:
                                            # Simple check: compare shapes first, then content if shapes match
                                            if last_saved_composite.shape == post_cropped_composite.shape:
                                                if np.array_equal(last_saved_composite, post_cropped_composite):
                                                    is_duplicate = True
                                        except Exception as e_comp:
                                             log = log_and_print(f"Warning: Error comparing images for duplication check: {e_comp}\n", log)
                                             
                                    if not is_duplicate:
                                        # Use helper to append RGBA correctly
                                        append_result(post_cropped_composite)
                                        # Update last_saved_composite only if append is successful
                                        if last_saved_composite is not None: del last_saved_composite
                                        last_saved_composite = post_cropped_composite.copy() # Store the saved one (BGR/BGRA)
                                        log = log_and_print(f"Saved composite image {len(stitched_results_rgba)} (Post-Cropped Shape: {post_cropped_composite.shape}).\n", log)
                                    else:
                                        log = log_and_print("Skipping save: Result identical to previously saved image.\n", log)
                                        
                                    # Clean up the post-cropped version if it wasn't stored in last_saved_composite
                                    if last_saved_composite is not post_cropped_composite:
                                        del post_cropped_composite
                                        gc.collect()
                                else:
                                    log = log_and_print("Warning: Post-stitch cropping failed for the size-limited composite, skipping save.\n", log)
                                    if post_cropped_composite is not None:
                                        del post_cropped_composite # Delete if it existed but was empty

                                # Reset for the next frame to become the anchor
                                del current_composite
                                if anchor_frame is not None:
                                    del anchor_frame # Delete old anchor too
                                if last_saved_composite is not None:
                                    del last_saved_composite # Reset duplicate check too
                                current_composite = None
                                anchor_frame = None
                                last_saved_composite = None
                                gc.collect()
                            # --- End Size Check ---

                        else:
                            # --- Stitching FAILED ---
                            log = log_and_print(f"Failed: Could not stitch frame {frame_num} onto current composite.\n", log)
                            # Save the *previous* valid composite (if it exists and is not just the anchor)
                            save_previous = False
                            if current_composite is not None and anchor_frame is not None:
                                # Check if composite is actually different from the anchor
                                try:
                                    if current_composite.shape != anchor_frame.shape or not np.array_equal(current_composite, anchor_frame):
                                        save_previous = True
                                except Exception as e_comp:
                                    log = log_and_print(f"Warning: Error comparing composite to anchor: {e_comp}\n", log)
                                    save_previous = True # Assume different if compare fails

                            if save_previous:
                                log = log_and_print("ACTION: Saving the previously stitched result before resetting.\n", log)
                                # Apply FINAL black border cropping if enabled
                                post_cropped_composite = crop_black_borders(current_composite, enable_cropping, strict_no_black_edges)
                                if post_cropped_composite is not None and post_cropped_composite.size > 0:
                                    # Check duplicate
                                    is_duplicate = False
                                    if last_saved_composite is not None and last_saved_composite.shape == post_cropped_composite.shape:
                                        if np.array_equal(last_saved_composite, post_cropped_composite):
                                            is_duplicate = True
                                
                                    if not is_duplicate:
                                        # Use helper to append RGBA correctly
                                        append_result(post_cropped_composite)
                                        if last_saved_composite is not None:
                                            del last_saved_composite
                                        last_saved_composite = post_cropped_composite.copy() # Store BGR/BGRA
                                        log = log_and_print(f"Saved composite image {len(stitched_results_rgb)} (Post-Cropped Shape: {post_cropped_composite.shape}).\n", log)
                                    else:
                                        log = log_and_print("Skipping save: Result identical to previously saved image.\n", log)
                                        
                                    if last_saved_composite is not post_cropped_composite:
                                        del post_cropped_composite
                                        gc.collect()
                                else:
                                    log = log_and_print("Warning: Post-stitch cropping failed for the previously stitched result, skipping save.\n", log)
                                    if post_cropped_composite is not None:
                                        del post_cropped_composite
                            else:
                                log = log_and_print("No previous composite to save (stitching failed on first attempt for this anchor or composite was just the anchor).\n", log)
                                
                            # The frame that *failed* to stitch becomes the new anchor
                            log = log_and_print(f"ACTION: Setting frame {frame_num} (shape: {current_frame_for_stitch.shape}) as the new anchor.\n", log)
                            if current_composite is not None:
                                del current_composite # Delete the old composite
                            if anchor_frame is not None:
                                del anchor_frame           # Delete the old anchor
                            if last_saved_composite is not None:
                                del last_saved_composite # Reset duplicate check
                            gc.collect()
                            # Reset anchor to current frame
                            anchor_frame = current_frame_for_stitch.copy() # Use the frame that failed (already cropped)
                            current_composite = anchor_frame # Reset composite to this new anchor
                            last_saved_composite = None
                            gc.collect()
                            # current_frame_for_stitch is now anchor_frame, no need to delete separately below

                    # Clean up current frame AFTER processing (if it wasn't made the new anchor)
                    # If stitching succeeded OR if it failed but wasn't the first frame,
                    # current_frame_for_stitch needs cleanup unless it just became the anchor.
                    if 'current_frame_for_stitch' in locals() and current_frame_for_stitch is not anchor_frame:
                        del current_frame_for_stitch
                        gc.collect()

                else: # Handle cases where frame_bgr_raw is None or empty after read
                    if frame_bgr_raw is not None:
                        del frame_bgr_raw
                        frame_bgr_raw = None
                    gc.collect()
            else: # Frame not sampled
                 # Still need to release the raw frame if it was read
                 if frame_bgr_raw is not None:
                     del frame_bgr_raw
                     frame_bgr_raw = None
                     # Don't gc.collect() on every skipped frame, too slow

            frame_num += 1
            # Loop continues
        except Exception as loop_error:
                log = log_and_print(f"Unexpected error in main video loop at frame {frame_num}: {loop_error}\n{traceback.format_exc()}\n", log)
                # Try to continue to next frame if possible, or break if capture seems broken
                if cap is None or not cap.isOpened():
                    log = log_and_print("Video capture likely broken, stopping loop.\n", log)
                    break
                else:
                    frame_num += 1 # Ensure frame counter increments
                    # Clean up potentially lingering frame data from the failed iteration
                    if 'frame_bgr_raw' in locals() and frame_bgr_raw is not None:
                        del frame_bgr_raw
                    if 'frame_bgr' in locals() and frame_bgra is not None:
                        del frame_bgra
                    if 'cropped_frame_bgr' in locals() and cropped_frame_bgra is not None:
                        del cropped_frame_bgra
                    if 'current_frame_for_stitch' in locals() and current_frame_for_stitch is not None and current_frame_for_stitch is not anchor_frame:
                        del current_frame_for_stitch
                    gc.collect()
                    
    # After the loop: Check if there's a final composite to save
    if current_composite is not None and anchor_frame is not None:
        # Only save if it contains more than just the last anchor frame OR if it's the *only* result
        save_final = False
        if len(stitched_results_rgba) == 0: # If no images saved yet, save this one
            save_final = True
        else:
            try:
                if current_composite.shape != anchor_frame.shape or not np.array_equal(current_composite, anchor_frame):
                    save_final = True
            except Exception as e_comp:
                log = log_and_print(f"Warning: Error comparing final composite to anchor: {e_comp}\n", log)
                save_final = True # Save if comparison fails
        
        if save_final:
            log = log_and_print("\nEnd of frames reached. Checking final composite...\n", log)
            post_cropped_final = crop_black_borders(current_composite, enable_cropping, strict_no_black_edges)
            if post_cropped_final is not None:
                 is_duplicate = False
                 if last_saved_composite is not None and last_saved_composite.shape == post_cropped_final.shape:
                     if np.array_equal(last_saved_composite, post_cropped_final):
                         is_duplicate = True
                 if not is_duplicate:
                     append_result(post_cropped_final) # RGBA append
                     log = log_and_print(f"Saved final composite image {len(stitched_results_rgba)} (Post-Cropped Shape: {post_cropped_composite.shape}).\n", log)
                     # No need to update last_saved_composite here, loop is finished
                 else:
                     log = log_and_print("Skipping save of final composite: Result identical to previously saved image.\n", log)
                 
                 # Clean up final cropped image if it existed
                 del post_cropped_final
                 gc.collect()
            else:
                log = log_and_print("Warning: Post-stitch cropping failed for the final composite, skipping save.\n", log)
                if post_cropped_final is not None:
                    del post_cropped_final # Delete if empty
        else:
            log = log_and_print("\nEnd of frames reached. Final composite was identical to its anchor frame and not the only result, not saving.\n", log)

    # --- Final Cleanup ---
    if cap is not None and cap.isOpened():
        cap.release()
    if 'cap' in locals():
        del cap
    if 'anchor_frame' in locals() and anchor_frame is not None:
        del anchor_frame
    if 'current_composite' in locals() and current_composite is not None:
        del current_composite
    if 'last_saved_composite' in locals() and last_saved_composite is not None:
        del last_saved_composite
    gc.collect()

    total_end_time = time.time()
    log = log_and_print(f"\nVideo stitching process finished. Found {len(stitched_results_rgb)} stitched image(s).", log)
    log = log_and_print(f"\nTotal processing time: {total_end_time - total_start_time:.2f} seconds.\n", log)

    # Filter out potential None entries just before returning
    final_results = [img for img in stitched_results_rgb if img is not None and img.size > 0]
    if len(final_results) != len(stitched_results_rgb):
        log = log_and_print(f"Warning: Filtered out {len(stitched_results_rgb) - len(final_results)} None or empty results before final return.\n", log)
        # Clean up the original list with potential Nones
        del stitched_results_rgb
        gc.collect()

    return final_results, log


# --- Gradio Interface Function ---
def run_stitching_interface(input_files,
    crop_top_percent,
    crop_bottom_percent,
    stitcher_mode_str, # For cv2.Stitcher
    registration_resol,
    seam_estimation_resol,
    compositing_resol,
    wave_correction,
    exposure_comp_type_str, # For cv2.Stitcher
    enable_cropping, # Post-stitch black border crop
    strict_no_black_edges_input,
    # Detailed Stitcher Settings
    transform_model_str,
    blend_method_str,
    enable_gain_compensation,
    orb_nfeatures,
    match_ratio_thresh,
    ransac_reproj_thresh_input,
    max_distance_coeff_input,
    max_blending_width,
    max_blending_height,
    blend_smooth_ksize_input,
    num_blend_levels_input,
    # Video specific settings
    sample_interval_ms,
    max_composite_width_video,
    max_composite_height_video,
    progress=gr.Progress(track_tqdm=True)
    ):
    """
    Wrapper function called by the Gradio interface.
    Handles input (images or video), applies pre-cropping,
    calls the appropriate stitching logic (passing transform_model_str),
    and returns results.
    Updates image reading and result saving to handle RGBA/BGRA transparency.
    UPDATED to handle transparency correctly when saving to temp files.
    """
    if input_files is None or len(input_files) == 0:
        return [], "Please upload images or a video file."
    
    # Convert Gradio inputs to correct types
    blend_smooth_ksize = int(blend_smooth_ksize_input) if blend_smooth_ksize_input is not None else -1
    num_blend_levels = int(num_blend_levels_input) if num_blend_levels_input is not None else 4
    ransac_reproj_thresh = float(ransac_reproj_thresh_input) if ransac_reproj_thresh_input is not None else 3.0
    max_distance_coeff = float(max_distance_coeff_input) if max_distance_coeff_input is not None else 0.5

    log = f"Received {len(input_files)} file(s).\n"
    log = log_and_print(f"Pre-Crop Settings: Top={crop_top_percent}%, Bottom={crop_bottom_percent}%\n", log)
    log = log_and_print(f"Post-Crop Black Borders: Enabled={enable_cropping}, Strict Edges={strict_no_black_edges_input}\n", log)
    # Log detailed settings including new ones
    log = log_and_print(f"Detailed Settings: Transform={transform_model_str}, Blend={blend_method_str}, GainComp={enable_gain_compensation}, ORB={orb_nfeatures}, Ratio={match_ratio_thresh}\n", log)
    log = log_and_print(f"Detailed Settings Cont'd: RANSAC Thresh={ransac_reproj_thresh}, MaxDistCoeff={max_distance_coeff}, MaxBlendW={max_blending_width}, MaxBlendH={max_blending_height}, SmoothKSize={blend_smooth_ksize}, MBLevels={num_blend_levels}\n", log)
    progress(0, desc="Processing Input...")

    # Determine input type: List of images or a single video
    is_video_input = False
    video_path = None
    image_paths = []
    
    # Check file types using mimetypes
    try:
        # Handle potential TempfileWrappers or string paths
        input_filepaths = []
        for f in input_files:
            if hasattr(f, 'name'): # Gradio File object
                input_filepaths.append(f.name)
            elif isinstance(f, str): # String path (e.g., from examples)
                input_filepaths.append(f)
            else:
                 log = log_and_print(f"Warning: Unexpected input file type: {type(f)}. Skipping.\n", log)

        if len(input_filepaths) == 1:
            filepath = input_filepaths[0]
            mime_type, _ = mimetypes.guess_type(filepath)
            if mime_type and mime_type.startswith('video'):
                is_video_input = True
                video_path = filepath
                log = log_and_print(f"Detected video input: {os.path.basename(video_path)}\n", log)
            elif mime_type and mime_type.startswith('image'):
                log = log_and_print("Detected single image input. Need at least two images for list stitching.\n", log)
                image_paths = [filepath] # Keep it for error message later
            else:
                # Fallback check: try reading as image
                img_test = None
                try:
                    # Use np.fromfile for paths that might have unicode characters
                    n = np.fromfile(filepath, np.uint8)
                    if n.size > 0:
                        img_test = cv2.imdecode(n, cv2.IMREAD_COLOR)
                    else:
                        raise ValueError("File is empty")

                    if img_test is not None and img_test.size > 0:
                        log = log_and_print(f"Warning: Unknown file type for single file: {os.path.basename(filepath)}. Assuming image based on successful read. Need >= 2 images.\n", log)
                        image_paths = [filepath]
                        del img_test
                    else:
                        raise ValueError("Cannot read as image or image is empty")
                except Exception as e_read_check:
                    log = log_and_print(f"Error: Could not determine file type or read single file: {os.path.basename(filepath)}. Error: {e_read_check}. Please provide video or image files.\n", log)
                    if img_test is not None:
                        del img_test
                    return [], log
        else: # Multiple files uploaded
            image_paths = []
            non_image_skipped = False
            for filepath in input_filepaths:
                mime_type, _ = mimetypes.guess_type(filepath)
                is_image = False
                if mime_type and mime_type.startswith('image'):
                    is_image = True
                else:
                    # Fallback check: Try reading as image
                    img_test = None
                    try:
                        n = np.fromfile(filepath, np.uint8)
                        if n.size > 0:
                            img_test = cv2.imdecode(n, cv2.IMREAD_COLOR)
                        else:
                            raise ValueError("File is empty")

                        if img_test is not None and img_test.size > 0:
                            is_image = True
                            log = log_and_print(f"Warning: Non-image or unknown file type detected: {os.path.basename(filepath)}. Assuming image based on read success.\n", log)
                            del img_test
                        else:
                            non_image_skipped = True
                            log = log_and_print(f"Warning: Skipping non-image file (or empty/failed read): {os.path.basename(filepath)}\n", log)
                    except Exception as e_read_check:
                        non_image_skipped = True
                        log = log_and_print(f"Warning: Skipping non-image file (read failed: {e_read_check}): {os.path.basename(filepath)}\n", log)
                        if img_test is not None:
                            del img_test


                if is_image:
                    image_paths.append(filepath)

            if not image_paths: # No valid images found
                if non_image_skipped:
                    log = log_and_print("Error: No valid image files found in the input list after filtering.\n", log)
                else: # Should only happen if initial list was empty, but covered by check at start
                    log = log_and_print("Error: No image files provided in the input list.\n", log)
                return [], log
            elif non_image_skipped:
                log = log_and_print(f"Proceeding with {len(image_paths)} assumed image files (some non-images were skipped).\n", log)
            else:
                log = log_and_print(f"Detected {len(image_paths)} image inputs.\n", log)
    except Exception as e:
         log = log_and_print(f"Error detecting input file types: {e}\n{traceback.format_exc()}\n", log)
         return [], log
     

    # --- Process Based on Input Type ---
    final_stitched_images_rgba = [] # This will technically hold RGBA if available
    stitch_log = ""

    if is_video_input:
        # --- VIDEO PROCESSING ---
        log = log_and_print("Starting incremental video frame stitching...\n", log)
        progress(0.1, desc="Sampling & Stitching Video...")
        # Ensure blend method string is lowercase for internal checks
        blend_method_lower = blend_method_str.lower() if blend_method_str else "multi-band"

        final_stitched_images_rgba, stitch_log = stitch_video_frames(
            video_path,
            crop_top_percent=crop_top_percent,
            crop_bottom_percent=crop_bottom_percent,
            enable_cropping=enable_cropping, # Post-stitch crop
            strict_no_black_edges=strict_no_black_edges_input,
            transform_model_str=transform_model_str,
            blend_method=blend_method_lower, # linear or multi-band
            enable_gain_compensation=enable_gain_compensation,
            orb_nfeatures=orb_nfeatures,
            match_ratio_thresh=match_ratio_thresh,
            ransac_reproj_thresh=ransac_reproj_thresh,
            max_distance_coeff=max_distance_coeff,
            max_blending_width=max_blending_width,
            max_blending_height=max_blending_height,
            sample_interval_ms=sample_interval_ms,
            max_composite_width=max_composite_width_video,
            max_composite_height=max_composite_height_video,
            blend_smooth_ksize=blend_smooth_ksize,
            num_blend_levels=num_blend_levels,
            progress=progress
        )

    elif len(image_paths) >= 2:
        # --- IMAGE LIST PROCESSING ---
        log = log_and_print("Reading and preparing images for list stitching...\n", log)
        images_bgr_cropped = [] # Store potentially cropped BGRA images
        read_success = True
        for i, img_path in enumerate(image_paths):
            progress(i / len(image_paths) * 0.1, desc=f"Reading image {i+1}/{len(image_paths)}") # Small progress for reading
            img = None
            try:
                n = np.fromfile(img_path, np.uint8)
                if n.size > 0:
                        # Use IMREAD_UNCHANGED to keep Alpha
                        img = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
                else:
                        log = log_and_print(f"Error: File is empty: {os.path.basename(img_path)}. Skipping.\n", log)
                        continue
                if img is None:
                         raise ValueError("imdecode returned None")
            except Exception as e_read:
                log = log_and_print(f"Error reading image: {os.path.basename(img_path)}. Error: {e_read}. Skipping.\n", log)
                if img is not None:
                    del img
                continue # Skip to the next image

            # Convert to BGRA for consistency
            img_bgra = None
            try:
                if img.ndim == 2:
                    img_bgra = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
                elif img.ndim == 3 and img.shape[2] == 3:
                    img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                elif img.ndim == 3 and img.shape[2] == 4:
                    img_bgra = img # Already BGRA
                else:
                    log = log_and_print(f"Error: Invalid image shape {img.shape} for {os.path.basename(img_path)}. Skipping.\n", log)
                    del img
                    if 'img_bgra' in locals() and img_bgra is not None:
                        del img_bgra
                    gc.collect()
                    continue # Skip to the next image
            except cv2.error as e_cvt_color:
                log = log_and_print(f"Error converting image color for {os.path.basename(img_path)}: {e_cvt_color}. Skipping.\n", log)
                del img
                if 'img_bgra' in locals() and img_bgra is not None:
                    del img_bgra
                gc.collect()
                continue

            # Release original read image if conversion happened
            if img_bgra is not img:
                del img
                gc.collect()

            # Apply Percentage Cropping
            img_bgra_cropped_single = crop_image_by_percent(img_bgra, crop_top_percent, crop_bottom_percent)

            # Release uncropped BGRA version
            if img_bgra_cropped_single is not img_bgra:
                del img_bgra
                gc.collect()

            if img_bgra_cropped_single is None or img_bgra_cropped_single.size == 0:
                log = log_and_print(f"Warning: Skipping image {os.path.basename(img_path)} because percentage cropping failed or resulted in empty image.\n", log)
                if img_bgra_cropped_single is not None:
                    del img_bgra_cropped_single
                gc.collect()
                continue # Skip to next image

            images_bgr_cropped.append(img_bgra_cropped_single)
            # log = log_and_print(f"Read and pre-cropped: {os.path.basename(img_path)} -> Shape: {img_bgra_cropped_single.shape}\n", log) # Can be verbose

        if len(images_bgr_cropped) < 2:
            stitch_log = log_and_print(f"Need at least two valid images after reading and pre-cropping ({len(images_bgr_cropped)} found) for list stitching.\n", log) 
            read_success = False # Indicate failure to proceed
        else:
            log = log_and_print(f"Proceeding with {len(images_bgr_cropped)} valid, pre-cropped images. Starting list stitching...\n", log)
            progress(0.1, desc="Stitching Image List...")
            # Ensure blend method string is lowercase for internal checks
            blend_method_lower = blend_method_str.lower() if blend_method_str else "multi-band"

            # Call the modified stitch_multiple_images function
        stitched_single_rgb, stitch_log_img = stitch_multiple_images(
            images_bgr_cropped, # Pass the list of cropped images (BGRA)
            stitcher_mode_str=stitcher_mode_str,
            registration_resol=registration_resol,
            seam_estimation_resol=seam_estimation_resol,
            compositing_resol=compositing_resol,
            wave_correction=wave_correction,
            exposure_comp_type_str=exposure_comp_type_str,
            enable_cropping=enable_cropping, # Post-stitch crop
            strict_no_black_edges=strict_no_black_edges_input,
            transform_model_str=transform_model_str,
            blend_method=blend_method_lower,
            enable_gain_compensation=enable_gain_compensation,
            orb_nfeatures=orb_nfeatures,
            match_ratio_thresh=match_ratio_thresh,
            ransac_reproj_thresh=ransac_reproj_thresh,
            max_distance_coeff=max_distance_coeff,
            max_blending_width=max_blending_width,
            max_blending_height=max_blending_height,
            blend_smooth_ksize=blend_smooth_ksize,
            num_blend_levels=num_blend_levels
        )
        stitch_log += stitch_log_img # Append log from stitching function
        if stitched_single_rgb is not None:
            final_stitched_images_rgba = [stitched_single_rgb] # Result is a list containing the single image

        # Clean up loaded images for list mode after stitching attempt
        if 'images_bgr_cropped' in locals():
            for img_del in images_bgr_cropped:
                if img_del is not None:
                    del img_del
            del images_bgr_cropped
            gc.collect()

    elif len(image_paths) == 1:
        # This case should have been handled by the input type detection,
        # but add a message here just in case.
        log = log_and_print("Error: Only one image file provided or detected. Need at least two for image list stitching.\n", log)
        stitch_log = "" # No stitching attempted
    else:
        # This case means no valid input files were found or passed initial checks.
        log = log_and_print("Error: Input must be a single video file or at least two image files. No valid input found.\n", log)
        stitch_log = ""

    final_log = log + stitch_log
    if not final_stitched_images_rgba:
        # Avoid duplicating error messages if log already indicates failure
        if "Error:" not in final_log[-200:]: # Check last few lines for errors
            final_log = log_and_print("\nNo stitched images were generated.", final_log)

    # Saving Results to Temporary Files (Handling RGBA)
    output_file_paths = [] # List to store paths for the Gallery
    temp_dir = None

    if final_stitched_images_rgba:
        try:
            # Try to create a subdirectory within the default Gradio temp space if possible
            gradio_temp_base = tempfile.gettempdir()
            gradio_subdir = os.path.join(gradio_temp_base, 'gradio') # Default Gradio temp subdir name
            # Check if we can write there, otherwise use default temp dir
            target_temp_dir_base = gradio_subdir if os.path.exists(gradio_subdir) and os.access(gradio_subdir, os.W_OK) else gradio_temp_base

            if not os.path.exists(target_temp_dir_base):
                try:
                    os.makedirs(target_temp_dir_base)
                except OSError as e_mkdir:
                    final_log = log_and_print(f"Warning: Could not create temp directory '{target_temp_dir_base}', using default. Error: {e_mkdir}\n", final_log)
                    target_temp_dir_base = tempfile.gettempdir() # Fallback to system default temp

            temp_dir = tempfile.mkdtemp(prefix="stitch_run_", dir=target_temp_dir_base)
            final_log = log_and_print(f"\nInfo: Saving output images to temporary directory: {temp_dir}\n", final_log)

            for i, img_rgb in enumerate(final_stitched_images_rgba):
                if img_rgb is None or img_rgb.size == 0:
                    final_log = log_and_print(f"Warning: Skipping saving image index {i} because it is None or empty.\n", final_log)
                    continue
                filename = f"stitched_image_{i+1:03d}.png" # PNG is required for transparency
                # Use os.path.join for cross-platform compatibility
                full_path = os.path.join(temp_dir, filename)
                img_bgr = None # Initialize for finally block
                try:
                    # Handle RGBA to BGRA for saving
                    # The result coming from stitcher is in RGB(A) format
                    if img_rgb.shape[2] == 4:
                        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2BGRA)
                    else:
                        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                
                    # Use imencode -> write pattern for better handling of paths/special chars
                    is_success, buf = cv2.imencode('.png', img_bgr)
                    if is_success:
                        with open(full_path, 'wb') as f:
                            f.write(buf)
                        # Use Gradio File obj or just path string? Gallery seems to prefer path strings.
                        output_file_paths.append((full_path, filename)) # Store the full path for Gradio Gallery
                        # final_log = log_and_print(f"Successfully saved: {filename}\n", final_log) # Can be verbose
                    else:
                     final_log = log_and_print(f"Warning: Failed to encode image for saving: {filename}\n", final_log)
                except cv2.error as e_cvt_write:
                    final_log = log_and_print(f"Error converting or encoding image {filename}: {e_cvt_write}\n", final_log)
                except IOError as e_io:
                    final_log = log_and_print(f"Error writing image file {filename} to {full_path}: {e_io}\n", final_log)
                except Exception as e_write:
                    final_log = log_and_print(f"Unexpected error writing image {filename} to {full_path}: {e_write}\n", final_log)
                finally:
                    if img_bgr is not None:
                        del img_bgr
                    gc.collect()
        except Exception as e_tempdir:
            final_log = log_and_print(f"Error creating temporary directory or saving output: {e_tempdir}\n", final_log)
            output_file_paths = [] # Fallback to empty list
            
    # --- Final Cleanup of RGB images list ---
    if 'final_stitched_images_rgb' in locals():
        for img_del in final_stitched_images_rgba:
            if img_del is not None:
                del img_del
        del final_stitched_images_rgba
        gc.collect()

    progress(1.0, desc="Finished!")
    final_log = log_and_print("\nCleanup complete.", final_log)

    # Return the LIST OF FILE PATHS for the Gallery, and the log
    return output_file_paths, final_log

# --- Define Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# OpenCV Image and Video Stitcher")
    gr.Markdown(
        "Upload multiple images (for list/panorama stitching) OR a single video file (for sequential frame stitching). "
        "Video frames are sampled incrementally based on the interval. "
        "Use Pre-Cropping to remove unwanted areas *before* stitching. Adjust other parameters and click 'Stitch'."
    )

    with gr.Row():
        with gr.Column(scale=1):
            stitch_button = gr.Button("Stitch", variant="primary")
            input_files = gr.File(
                label="Upload Images or a Video",
                # Common image and video types
                file_types=["image", ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"],
                file_count="multiple",
                elem_id="input_files"
            )

            # --- Parameters Grouping ---
            with gr.Accordion("Preprocessing Settings", open=True):
                crop_top_percent = gr.Slider(0.0, 49.0, step=0.5, value=0.0, label="Crop Top %",
                                                                         info="Percentage of height to remove from the TOP of each image/frame BEFORE stitching.")
                crop_bottom_percent = gr.Slider(0.0, 49.0, step=0.5, value=0.0, label="Crop Bottom %",
                                                                                info="Percentage of height to remove from the BOTTOM of each image/frame BEFORE stitching.")

            with gr.Accordion("OpenCV Stitcher Settings (Image List Mode Only)", open=True):
                stitcher_mode = gr.Radio(["SCANS", "PANORAMA", "DIRECT_PAIRWISE"], label="Stitcher Mode (Image List)", value="SCANS",
                    info=(
                        "Method for image list stitching. 'SCANS'/'PANORAMA': Use OpenCV's built-in Stitcher (optimized for translation/rotation). "
                        "'SCANS': Optimized for images primarily related by translation (like scanning documents or linear camera motion), potentially using simpler geometric models or assumptions internally. "
                        "'PANORAMA': Designed for images captured by rotating the camera around a central point. It uses full perspective transformations (Homography) to handle the complex geometric distortions typical in panoramic shots."
                        "'DIRECT_PAIRWISE': Skip OpenCV Stitcher and directly use sequential pairwise feature matching (same as video mode or fallback)."
                    )
                )
                registration_resol = gr.Slider(0.1, 1.0, step=0.05, value=0.6, label="Registration Resolution",
                                                                             info="Scale factor for the image resolution used during feature detection and matching. Lower values (e.g., 0.6) are faster but may miss features in high-res images. 1.0 uses full resolution.")
                seam_estimation_resol = gr.Slider(0.05, 1.0, step=0.05, value=0.1, label="Seam Estimation Resolution",
                                                                                    info="Scale factor for the image resolution used during seam finding (finding the optimal cut line). Lower values (e.g., 0.1) are much faster.")
                compositing_resol = gr.Slider(-1.0, 1.0, step=0.1, value=-1.0, label="Compositing Resolution",
                                                                            info="Scale factor for the image resolution used during the final blending stage. -1.0 uses the original source image resolution. Lower values reduce memory usage but might slightly blur the output.")
                wave_correction = gr.Checkbox(value=False, label="Enable Wave Correction",
                                                                            info="Attempts to correct perspective distortions (waviness) common in panoramas. Can increase processing time.")
                exposure_comp_type = gr.Dropdown(["NO", "GAIN", "GAIN_BLOCKS"], value="GAIN_BLOCKS", label="Exposure Compensation",
                                                                                 info="Method used by the built-in stitcher to adjust brightness/contrast differences between images. 'GAIN_BLOCKS' is generally preferred for varying lighting.")

            # --- Detailed Stitcher Settings (Used for Video, DIRECT_PAIRWISE, and Fallback) ---
            with gr.Accordion("Pairwise Stitching Settings (Video / Direct / Fallback)", open=True):
                transform_model = gr.Radio(["Homography", "Affine_Partial", "Affine_Full"], label="Pairwise Transform Model", value="Homography", # Default to Homography
                                                                     info="Geometric model for pairwise alignment. 'Homography' handles perspective. 'Affine' (Partial/Full) handles translation, rotation, scale, shear (better for scans, less distortion risk). If stitching fails with one model, try another.")
                blend_method = gr.Radio(["Linear", "Multi-Band"], label="Blending Method", value="Multi-Band",
                                                                info="Algorithm for smoothing seams in overlapping regions when using the detailed stitcher (for video or image list fallback). 'Multi-Band' is often better but slower.")
                enable_gain_compensation = gr.Checkbox(value=True, label="Enable Gain Compensation",
                                                                                                info="Adjusts overall brightness difference *before* blending when using the detailed stitcher. Recommended.")
                orb_nfeatures = gr.Slider(500, 10000, step=100, value=2000, label="ORB Features",
                                                                     info="Maximum ORB keypoints detected per image/frame. Used by the detailed stitcher (for video or image list fallback).")
                match_ratio_thresh = gr.Slider(0.5, 0.95, step=0.01, value=0.75, label="Match Ratio Threshold",
                                                                             info="Lowe's ratio test threshold for filtering feature matches (lower = stricter). Used by the detailed stitcher (for video or image list fallback).")
                ransac_reproj_thresh = gr.Slider(1.0, 10.0, step=0.1, value=5.0, label="RANSAC Reproj Threshold",
                                                                                info="Maximum reprojection error (pixels) allowed for a match to be considered an inlier by RANSAC during transformation estimation. Lower values are stricter.")
                max_distance_coeff = gr.Slider(0.1, 2.0, step=0.05, value=0.5, label="Max Distance Coeff",
                                                                             info="Multiplier for image diagonal used to filter initial matches. Limits the pixel distance between matched keypoints (0.5 means half the diagonal).")
                max_blending_width = gr.Number(value=10000, label="Max Blending Width", precision=0,
                                                                             info="Limits the canvas width during the detailed pairwise blending step to prevent excessive memory usage. Relevant for the detailed stitcher.")
                max_blending_height = gr.Number(value=10000, label="Max Blending Height", precision=0,
                                                                                info="Limits the canvas height during the detailed pairwise blending step to prevent excessive memory usage. Relevant for the detailed stitcher.")
                blend_smooth_ksize = gr.Number(value=15, label="Blend Smooth Kernel Size", precision=0,
                                                                             info="Size of Gaussian kernel to smooth blend mask/weights. Must be POSITIVE ODD integer to enable smoothing (e.g., 5, 15, 21). Set to -1 or an even number to disable smoothing.")
                num_blend_levels = gr.Slider(2, 7, step=1, value=4, label="Multi-Band Blend Levels",
                                                                         info="Number of pyramid levels for Multi-Band blending. Fewer levels are faster but might have less smooth transitions.")

            with gr.Accordion("Video Stitcher Settings", open=False):
                sample_interval_ms = gr.Number(value=3000, label="Sample Interval (ms)", precision=0,
                                                                             info="Time interval (in milliseconds) between sampled frames for video stitching. Smaller values sample more frames, increasing processing time but potentially improving tracking.")
                max_composite_width_video = gr.Number(value=10000, label="Max Composite Width (Video)", precision=0,
                                                                                         info="Limits the width of the stitched output during video processing. If exceeded, the current result is saved and stitching restarts with the next frame. 0 = no limit.")
                max_composite_height_video = gr.Number(value=10000, label="Max Composite Height (Video)", precision=0,
                                                                                            info="Limits the height of the stitched output during video processing. If exceeded, the current result is saved and stitching restarts with the next frame. 0 = no limit.")

            with gr.Accordion("Postprocessing Settings", open=False):
                enable_cropping = gr.Checkbox(value=True, label="Crop Black Borders (Post-Stitch)",
                                                                            info="Automatically remove black border areas from the final stitched image(s) AFTER stitching.")
                strict_no_black_edges_checkbox = gr.Checkbox(value=False, label="Strict No Black Edges (Post-Crop)",
                    info="If 'Crop Black Borders' is enabled, this forces removal of *any* remaining black pixels directly on the image edges after the main crop. Might slightly shrink the image further.")

        with gr.Column(scale=1):
            output_gallery = gr.Gallery(
                label="Stitched Results", elem_id="output_gallery", object_fit="contain", type="filepath", rows=2, preview=True, height="auto", format="png", container=True)
            output_log = gr.Textbox(
                label="Status / Log", lines=20, interactive=False, show_copy_button=True)


        # Define the list of inputs for the button click event
        inputs=[
                input_files,
                # Preprocessing
                crop_top_percent,
                crop_bottom_percent,
                # OpenCV Stitcher (Image List)
                stitcher_mode, # the selected string ("SCANS", "PANORAMA", or "DIRECT_PAIRWISE")
                registration_resol,
                seam_estimation_resol,
                compositing_resol,
                wave_correction,
                exposure_comp_type,
                # Postprocessing
                enable_cropping,
                strict_no_black_edges_checkbox,
                # Detailed Stitcher Settings
                transform_model,
                blend_method,
                enable_gain_compensation,
                orb_nfeatures,
                match_ratio_thresh,
                ransac_reproj_thresh,
                max_distance_coeff,
                max_blending_width,
                max_blending_height,
                blend_smooth_ksize,
                num_blend_levels,
                # Video specific settings
                sample_interval_ms,
                max_composite_width_video,
                max_composite_height_video
         ]

        # Define examples (update to include the new transform_model parameter)
        examples = [
        [
            ["examples/Wetter-Panorama/Wetter-Panorama1[NIuO6hrFTrg].mp4"],
            0, 20,
            "DIRECT_PAIRWISE", 0.6, 0.1, -1, False, "GAIN_BLOCKS",
            True, False,
            "Homography", "Multi-Band", True, 5000, 0.5, 5.0, 0.5, 10000, 10000, 15, 4,
            2500, 10000, 10000,
        ],
        [
            ["examples/Wetter-Panorama/Wetter-Panorama2[NIuO6hrFTrg].mp4"],
            0, 20,
            "DIRECT_PAIRWISE", 0.6, 0.1, -1, False, "GAIN_BLOCKS",
            True, False,
            "Homography", "Multi-Band", True, 5000, 0.5, 5.0, 0.5, 10000, 10000, 15, 4,
            2500, 10000, 10000,
        ],
        [
            ["examples/NieRAutomata/nier2B_01.jpg", "examples/NieRAutomata/nier2B_02.jpg", "examples/NieRAutomata/nier2B_03.jpg", "examples/NieRAutomata/nier2B_04.jpg", "examples/NieRAutomata/nier2B_05.jpg",
             "examples/NieRAutomata/nier2B_06.jpg", "examples/NieRAutomata/nier2B_07.jpg", "examples/NieRAutomata/nier2B_08.jpg", "examples/NieRAutomata/nier2B_09.jpg", "examples/NieRAutomata/nier2B_10.jpg", ],
            0, 0,
            "PANORAMA", 0.6, 0.1, -1, False, "GAIN_BLOCKS",
            True, False,
            "Homography", "Multi-Band", True, 5000, 0.5, 5.0, 0.5, 10000, 10000, 15, 4,
            5000, 10000, 10000,
        ],
        [
            ["examples/cat/cat_left.jpg", "examples/cat/cat_right.jpg"],
            0, 0,
            "SCANS", 0.6, 0.1, -1, False, "GAIN_BLOCKS",
            True, False,
            "Affine_Partial", "Linear", True, 5000, 0.5, 5.0, 0.5, 10000, 10000, 15, 4,
            5000, 10000, 10000,
        ],
        [
            ["examples/ギルドの受付嬢ですが/Girumasu_1.jpg", "examples/ギルドの受付嬢ですが/Girumasu_2.jpg", "examples/ギルドの受付嬢ですが/Girumasu_3.jpg"],
            0, 0,
            "PANORAMA", 0.6, 0.1, -1, False, "GAIN_BLOCKS",
            True, False,
            "Affine_Partial", "Linear", True, 5000, 0.65, 5.0, 0.5, 10000, 10000, 15, 4,
            5000, 10000, 10000,
        ],
        [
            ["examples/photographs1/img1.jpg", "examples/photographs1/img2.jpg", "examples/photographs1/img3.jpg", "examples/photographs1/img4.jpg"],
            0, 0,
            "PANORAMA", 0.6, 0.1, -1, True, "GAIN_BLOCKS",
            True, False,
            "Homography", "Linear", True, 5000, 0.5, 5.0, 0.5, 10000, 10000, 15, 4,
            5000, 10000, 10000,
        ]
    ]
    gr.Examples(examples, inputs=inputs, label="Example Configurations")

    # Connect button click to the function
    stitch_button.click(
        fn=run_stitching_interface,
        inputs=inputs,
        outputs=[output_gallery, output_log]
    )

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting Gradio interface with selectable transformation model...")
    # Enable queue for handling multiple requests and progress updates
    demo.queue()
    # Launch the interface
    demo.launch(inbrowser=True)
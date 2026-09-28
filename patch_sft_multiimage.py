"""Always pass per-sample image LISTS to the processor (mixed nesting is rejected)."""
src = open('src/train/sft.py').read()
old = "                all_images.append(loaded_images[0] if len(loaded_images) == 1 else loaded_images)"
new = "                all_images.append(loaded_images)  # uniform nesting; processor flattens"
if new in src:
    print("already patched"); raise SystemExit
assert old in src, "ANCHOR NOT FOUND - paste me: grep -n 'all_images.append' src/train/sft.py"
open('src/train/sft.py', 'w').write(src.replace(old, new))
print("PATCHED OK: uniform image nesting")

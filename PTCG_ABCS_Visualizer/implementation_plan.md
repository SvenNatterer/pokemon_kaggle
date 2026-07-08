# Update Visualizer Layout to Match Screenshot

Based on the official Kaggle Visualizer screenshot, we need to adjust the sizing and positioning of elements on the playmat, and add damage indicators and energy symbols.

## Proposed Changes

### 1. Board Layout Adjustments
- **P0 (Bottom) Player**:
  - **Bench**: Moved to the **left** side of the playmat, stacked vertically (overlapping).
  - **Prizes**: Dashes positioned horizontally in the bottom-middle.
  - **Stats**: Moved to the **right** side of the playmat.
- **P1 (Top) Player**:
  - **Bench**: Moved to the **right** side of the playmat, stacked vertically.
  - **Prizes**: Dashes positioned horizontally on the **left** side.
  - **Stats**: Moved to the **left** side of the playmat, below their prizes.

### 2. Card Sizes
- **Active Cards**: Slightly smaller than currently (they are currently 3x size, which is too big). We'll scale them to match the screenshot proportion.
- **Bench/Hand Cards**: Keep base size, but ensure bench cards overlap correctly vertically (`margin-top` logic).

### 3. Damage Indicators
- Render the current damage (`c.damage`) on the cards as a large, bold red number with a black outline, positioned at the top-left of the card.

### 4. Attached Energy Symbols
- Render attached energies on the cards as small energy type icons, positioned at the top-right of the card. 

### 5. Playmat Aesthetics
- Change the center Pokeball from a solid/semi-transparent filled circle to just a white border outline, matching the clean look of the official visualizer.

## Verification
- Reload `index.html` and verify the layout, damage text, and energy symbols match the screenshot perfectly.

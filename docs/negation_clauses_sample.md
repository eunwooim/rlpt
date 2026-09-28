# VisualPRM negation pairs — clause-level view (sample of 20 / 10,000)

For each example: the `edit` applied to make `neg`, and the clause pair(s) that differ.
Clauses produced by the SAME `split_clauses` used in the experiment (newline + `.;:`).

---
### idx 0  |  source `mavis_function_poly_prm.jsonl`  |  edit `2->3`  |  17 clauses, 1 differ
**Q:** Displayed is the graph of f(x) = a*x - 1, under the constraint that a is non-zero. For x=2, the function value is 3. Taking into account the graph and the conditions given, what is the derivative value at x = 2?

- clause 2:
  - POS: `and the value of the function at \( x = 2 \) is 3:`
  - NEG: `and the value of the function at \( x = 3 \) is 3:`
---
### idx 1  |  source `geo170k_extracted_full_prm.jsonl`  |  edit `6.0->7`  |  23 clauses, 1 differ
**Q:** In the practical activity class, Xiao Ming constructed a conical funnel using cardboard. If the bottom radius of the funnel is denoted by variable 'x' (x = 6.0) and the height is denoted by variable 'y' (y = 8.0), what is the lateral area of the conical funnel? Express the area as 'z' in terms of 'x

- clause 4:
  - POS: `Given in the problem, the bottom radius of the funnel is \( x = 6.0 \) and the height of the cone is \( y = 8.0 \).`
  - NEG: `Given in the problem, the bottom radius of the funnel is \( x = 7 \) and the height of the cone is \( y = 8.0 \).`
---
### idx 2  |  source `CLEVR_math_en_20240402_extracted_prm.jsonl`  |  edit `9->10`  |  22 clauses, 1 differ
**Q:** Add 1 large red metal things. How many objects exist?

- clause 19:
  - POS: `8 (existing objects) + 1 (large red metal cube) = 9`
  - NEG: `8 (existing objects) + 1 (large red metal cube) = 10`
---
### idx 3  |  source `unigeo_calc_en_20240402_extracted_open_ended_only_prm.jsonl`  |  edit `50->51`  |  11 clauses, 1 differ
**Q:** 如图,在圆内接四边形ABCD中,∠B=50°,则∠D=()

- clause 1:
  - POS: `- ∠B = 50°`
  - NEG: `- ∠B = 51°`
---
### idx 4  |  source `geoqa+_extracted_en_version_prm.jsonl`  |  edit `120->121`  |  24 clauses, 1 differ
**Q:** In the rhombus ABCD, the diagonals AC and BD intersect at point O, ∠BAD = 120°, and AC = 4. The area of the rhombus is ( )
A. 16√{3}
B. 16
C. 8√{3}
D. 8

- clause 1:
  - POS: `Use the given ∠BAD = 120° to find ∠AOB:`
  - NEG: `Use the given ∠BAD = 121° to find ∠AOB:`
---
### idx 5  |  source `geoqa+_en_20240402_extracted_open_ended_only_prm.jsonl`  |  edit `5->6`  |  25 clauses, 1 differ
**Q:** 如图，在△ABC中，点D是BC边上的一点，E，F分别是AD，BE的中点，连接CE，CF，若S△CEF＝5，则△ABC的面积为（）

- clause 21:
  - POS: `△ABC的面积 = 5 + 5 = 10`
  - NEG: `△ABC的面积 = 6 + 5 = 10`
---
### idx 6  |  source `geomverse_extracted_prm.jsonl`  |  edit `84->85`  |  40 clauses, 1 differ
**Q:** If the ABCDE shape is a rectangle where an equilateral triangle has been removed from one side of it, the length of the BC side is 6 and the area of the ABCDE shape is 84, compute the length of the AB side of the ABCDE shape. Round computations to 2 decimal places.

- clause 18:
  - POS: `w \times 6 - \frac{\sqrt{3}}{4} s^2 = 84`
  - NEG: `w \times 6 - \frac{\sqrt{3}}{4} s^2 = 85`
---
### idx 7  |  source `geometry3k_en_20240402_extracted_open_ended_only_prm.jsonl`  |  edit `2->3`  |  16 clauses, 1 differ
**Q:** Find the measure of S T of equilateral triangle R S T if R S = x + 9, S T = 2 x, and R T = 3 x - 9.

- clause 3:
  - POS: `\[ x + 9 = 2x \]`
  - NEG: `\[ x + 9 = 3x \]`
---
### idx 8  |  source `unigeo_calc_en_20240402_extracted_open_ended_only_prm.jsonl`  |  edit `40->41`  |  5 clauses, 1 differ
**Q:** 如图,O为圆心,AB是直径,C是半圆上的点,D是\wide{AC}上的点．若∠BOC=40°,则∠D的大小为()

- clause 0:
  - POS: `解:∵OA=OD,∴∠A=∠AOD,∵∠BOC=∠AOD+∠D,∴40°=∠A+∠D,则2∠D=40°,`
  - NEG: `解:∵OA=OD,∴∠A=∠AOD,∵∠BOC=∠AOD+∠D,∴40°=∠A+∠D,则2∠D=41°,`
---
### idx 9  |  source `MathV360K_prompts_prm.jsonl`  |  edit `52->53`  |  14 clauses, 1 differ
**Q:** What is the range of the number of boxes sold?

- clause 9:
  - POS: `- Range = 52 - 29`
  - NEG: `- Range = 53 - 29`
---
### idx 10  |  source `mavis_function_poly_prm.jsonl`  |  edit `25->26`  |  40 clauses, 1 differ
**Q:** The figure shows the graph of function f(x) = a*x**2 + b*x + c, where a is non-zero.  The function's value when x=5 is 25. The derivative f'(x) takes the values 2 and 6 at x=1 and x=3, respectively. With the given graph and conditions, state the minimum values of the function.

- clause 2:
  - POS: `- \( f(5) = 25 \)`
  - NEG: `- \( f(5) = 26 \)`
---
### idx 11  |  source `unigeo_calc_en_20240402_extracted_open_ended_only_prm.jsonl`  |  edit `76->77`  |  13 clauses, 1 differ
**Q:** 如图,点A、B、C都在⊙O上,若∠AOB=76°,则∠ACB的度数为()

- clause 2:
  - POS: `\( \angle AOB = 76^\circ \)。`
  - NEG: `\( \angle AOB = 77^\circ \)。`
---
### idx 12  |  source `mavis_function_poly_prm.jsonl`  |  edit `2->3`  |  34 clauses, 1 differ
**Q:** The graph depicted is for the function f(x) = b*x + 4*x**2. The function f(x) takes the value 16 at x=2. Based on the provided graph and the given conditions, what is the value of f'(6)?

- clause 3:
  - POS: `We are told that the function takes the value 16 at \( x = 2 \).`
  - NEG: `We are told that the function takes the value 16 at \( x = 3 \).`
---
### idx 13  |  source `MathV360K_prompts_prm.jsonl`  |  edit `2->3`  |  13 clauses, 1 differ
**Q:** Fill in the blank to describe the model. The model has 6 dots divided into 3 equal groups. There are (_) dots in each group.

- clause 10:
  - POS: `- So, 6 dots / 3 groups = 2 dots per group.`
  - NEG: `- So, 6 dots / 3 groups = 3 dots per group.`
---
### idx 14  |  source `mavis_function_poly_prm.jsonl`  |  edit `2->3`  |  40 clauses, 1 differ
**Q:** Displayed is the graph of f(x) = a*x**2 + c + 4*x, under the constraint that a is non-zero. x=0 is a zero of the function f(x). The derivative value at x=6 is -8. With the graph and above conditions in mind, identify the maximum points of the function.

- clause 7:
  - POS: `f'(x) = \frac{d}{dx} (a * x^2 + c + 4x) = 2ax + 4`
  - NEG: `f'(x) = \frac{d}{dx} (a * x^2 + c + 4x) = 3ax + 4`
---
### idx 15  |  source `mavis_function_cos_prm.jsonl`  |  edit `-1->1`  |  35 clauses, 1 differ
**Q:** The graph of the function f(x) = a*cos(b*x + c) is shown, where the condition that a, b are positive, and c is in the range of (0, pi] is satisfied. The minimum values of function f(x) are achieved at f(8*k + 2/3)=-8, where k is an integer. find the derivative value at x = 2/3.

- clause 3:
  - POS: `We know that the minima of the cosine function occur at points where \( \cos(\theta) = -1 \).`
  - NEG: `We know that the minima of the cosine function occur at points where \( \cos(\theta) = 1 \).`
---
### idx 16  |  source `CLEVR_math_en_20240402_extracted_prm.jsonl`  |  edit `3->4`  |  13 clauses, 1 differ
**Q:** Add 1 large cylinders. How many large cylinders are left?

- clause 10:
  - POS: `Adding 1 large cylinder makes it 2 + 1 = 3 large cylinders.`
  - NEG: `Adding 1 large cylinder makes it 2 + 1 = 4 large cylinders.`
---
### idx 17  |  source `CLEVR_math_en_20240402_extracted_prm.jsonl`  |  edit `3->4`  |  17 clauses, 1 differ
**Q:** Subtract 2 cylinders. How many cylinders are left?

- clause 14:
  - POS: `So, 5 - 2 = 3 cylinders remain.`
  - NEG: `So, 5 - 2 = 4 cylinders remain.`
---
### idx 18  |  source `MathV360K_prompts_prm.jsonl`  |  edit `34->35`  |  17 clauses, 1 differ
**Q:** How many children painted an even number of flower pots?

- clause 13:
  - POS: `\[ 12 + 7 + 15 = 34 \]`
  - NEG: `\[ 12 + 7 + 15 = 35 \]`
---
### idx 19  |  source `geomverse_extracted_prm.jsonl`  |  edit `31.5->32.5`  |  8 clauses, 1 differ
**Q:** Compute the area of the purple right triangle. Round computations to 2 decimal places.

- clause 4:
  - POS: `\[ \text{Area} = 31.5 \]`
  - NEG: `\[ \text{Area} = 32.5 \]`

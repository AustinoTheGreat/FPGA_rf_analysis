
 
**README for final_part_b (Bitstream and RF Comparison and Analysis)**<br />
<br />
This code takes in one to four sets of rf and bistream data, performs analysis, visualize comparisons, and calculates JTAG frequency. <br />
It incorporates aspects such as bit transitions and energy signature and builds further comparison analysis off of them.  <br />

****User Directions****<br />
Please enter "final_part_b" directory and run "python3 main.py" <br />
Please find below what user should input with each question prompt below: <br />
<br />
Question Prompt: "Number of datasets to categorize: "<br />
Acceptable User Input: "1", "2", "3", or "4"<br />
Directions: 1 refers to using one set of rf data and bitstream for visualization. **Enter 4 for maximum comparison and analysis**<br />
<br />
Question Prompt: "Import complete, choose between a(peaks), b(JTAG freq.), c(envelop), q(quit): "<br />
Acceptable User Input: "a", "b", "c", "q"<br />
Directions:<br />

- "a": Visualize peaks for energy signatures and bitstream flips side-by-side for all input data sets<br />
- "b": Outputs reverse engineered JTAG freqeuncy of each input rf signatures<br />
- "c": Visualize top envelop signal for one chosen energy signature and one chosen bitstream flips for side-by-side comparison<br />
- "q": Quits the program<br />

<br />
Question Prompt: "Pick a rf energy signal for comparison (0-3, when 4 datasets were orginally imported): "<br />
Acceptable User Input: "0", "1", "2", "3" (when originally chosed 4 sets of data to categorize, otherwise n-1 options)<br />
Directions: Chose desired rf energy signature for visualization/comparison<br />
<br />
Question Prompt: "Pick a bitstream for comparison (0-3, when 4 datasets were orginally imported): "<br />
Acceptable User Input: "0", "1", "2", "3" (when originally chosed 4 sets of data to categorize, otherwise n-1 options)<br />
Directions: Chose desired bitstream flips signature for visualization/comparison<br />
<br />
Test Example: for last two questions, chose "3" for both to see visualization/comparison for 81.45% FPGA LUT utilization<br />
<br />

****IMPORTANT****<br />
Please note that it may take sometime for code to run earlier on when collecting all the data from files. Please be patient. Thank you.<br />
<br />

****How code works****<br />
Please see brief summary of how code works for options to question:<br />
"Import complete, choose between a(peaks), b(JTAG freq.), c(envelop), q(quit): "<br />

- "a": Peaks are calculated with threshold of 1.5 standard deviation above mean, corresponding to 87% percentile of data<br />
- "b": Input JTAG freqeuncy is calculated from a linear relationship, please find equation in code "main.py" line 434<br />
- "c": Upper envelop calculations are specified in code "main.py" starting at line 342<br />
<br />

****Outputs****<br />

- RF energy and bitstream flips graphs<br />
- Peaks of RF energy and bitstream flips graphs<br />
- JTAG frequencies<br />
- Upper envelop signal of RF energy and bitstream flips graphs<br />

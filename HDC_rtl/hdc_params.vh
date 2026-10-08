// hdc_params.vh

// HDC Core Parameters (from your experiments)
`define DIM             500
`define N_LEVELS        20
`define N_FEAT          14
`define N_MEMBERS       1        // ensemble=1
`define N_CLASSES       2        // CHF vs NSR

// Bit widths
`define LEVEL_BITS      5        // ceil(log2(20))
`define FEAT_BITS       4        // ceil(log2(14))
`define DIM_BITS        10       // greater than ceil(log2(500))
`define MEMBER_BITS     1        // 0 or 1 (since ensemble=1)
`define CLASS_BITS      8        // 8-bit class memory

// Quantization (matches Python's SCALE)
`define THRESH_W        16       // Q6.10 format 

// Similarity accumulator width
`define SIM_W           32       
`define ACC_W           32       

// Feature sum width (14 features, each ±1)
`define FEAT_SUM_W      5        

// Timing constants
`define N_COEF          14       // CA4 coefficients
`define FE_WINDOW_CYCLES 16*14 + 48  // Cycles for FE block

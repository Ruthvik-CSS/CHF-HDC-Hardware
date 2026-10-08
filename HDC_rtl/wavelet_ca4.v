// =============================================================================
// wavelet_ca4.v
//
// Folded, multiplier-less microarchitecture for the closed-form level-4
// quadratic-spline (QS) wavelet approximation coefficient CA4[n], Eq. (2) of:
//   Bhardwaj, Janveja, Krishnaswamy, Pidanic, Trivedi, "Design of Random
//   Forest-Based Low-Power VLSI Architecture to Detect Congestive Heart
//   Failure for Wearable Devices," IEEE TVLSI, 2026.
//
// Implements the Wavelet Control Unit (WCU) + Computational Module (CM)
// described in Fig. 5 / Section III-B:
//   - One ECG sample arrives per clock cycle (sample_valid / sample_in).
//   - Three folded accumulator "slots" (reg1/reg2/reg3 in the paper) run
//     concurrently. A new coefficient accumulation starts every 16 cycles,
//     round-robin over the 3 slots (coefficient n always uses slot n%3),
//     matching the overlapped W0/W1/W2 timing in Table II.
//   - Each slot applies the ramp-up(1..8)/hold(8)/ramp-down(7..1, last two
//     cycles dropped) scaling schedule of Eq. (2), using a shift-add
//     constant multiplier (k*x for k=0..8) instead of a general multiplier,
//     per the paper's "shift-based arithmetic in place of general-purpose
//     multipliers" design choice (Section III-B-3).
//   - After 48 cycles a slot's accumulation is complete; the paper's ">>4"
//     is applied and the result is written into the 14-entry coefficient
//     buffer (the "sliding buffer" of Fig. 5).
//
// =============================================================================
`timescale 1ns/1ps

module wavelet_ca4 #(
    parameter integer DW      = 16,
    parameter integer ACC_W   = 26,  
    parameter integer N_COEF  = 14   
) (
    input  wire                  clk,
    input  wire                  rst_n,

    input  wire                  sample_valid,
    input  wire signed [DW-1:0]  sample_in,

    output reg                   features_valid,
    output wire [N_COEF*DW-1:0]  coeff_out_flat
);

    localparam integer DRIVE_CYCLES = 16*(N_COEF-1) + 48;
    localparam integer CW            = $clog2(DRIVE_CYCLES+1);
    localparam integer NW            = $clog2(N_COEF+1);

    // ---------------------------------------------------------------
    // Global window counters
    // ---------------------------------------------------------------
    reg [CW-1:0] sample_counter;
    reg [NW-1:0] next_coeff;

    wire start_pulse = sample_valid && (next_coeff < N_COEF) && ((sample_counter % 16) == 0);

    function [1:0] slot_of_coeff(input [NW-1:0] n);
        slot_of_coeff = n % 3;
    endfunction

    function [3:0] weight_of(input [1:0] phase, input [3:0] pos);
        case (phase)
            2'd0:    weight_of = (pos >> 1) + 4'd1;
            2'd1:    weight_of = 4'd8;
            2'd2:    weight_of = (pos < 14) ? (4'd7 - (pos >> 1)) : 4'd0;
            default: weight_of = 4'd0;
        endcase
    endfunction

    function signed [ACC_W-1:0] wmul(input signed [DW-1:0] x, input [3:0] w);
        reg signed [ACC_W-1:0] xs;
        begin
            xs = {{(ACC_W-DW){x[DW-1]}}, x};
            case (w)
                4'd0: wmul = {ACC_W{1'b0}};
                4'd1: wmul = xs;
                4'd2: wmul = xs <<< 1;
                4'd3: wmul = (xs <<< 1) + xs;
                4'd4: wmul = xs <<< 2;
                4'd5: wmul = (xs <<< 2) + xs;
                4'd6: wmul = (xs <<< 2) + (xs <<< 1);
                4'd7: wmul = (xs <<< 3) - xs;
                4'd8: wmul = xs <<< 3;
                default: wmul = {ACC_W{1'b0}};
            endcase
        end
    endfunction

    function signed [DW-1:0] sat_dw(input signed [ACC_W-1:0] v);
        localparam signed [ACC_W-1:0] MAXV = (1 <<< (DW-1)) - 1;
        localparam signed [ACC_W-1:0] MINV = -(1 <<< (DW-1));
        begin
            if (v > MAXV)      sat_dw = MAXV[DW-1:0];
            else if (v < MINV) sat_dw = MINV[DW-1:0];
            else               sat_dw = v[DW-1:0];
        end
    endfunction

    // ---------------------------------------------------------------
    // Per-slot state (the 3 folded accumulator datapaths: reg1/reg2/reg3)
    // ---------------------------------------------------------------
    reg                    slot_active [0:2];
    reg  [1:0]             slot_phase  [0:2]; // 0=ramp-up, 1=hold@8, 2=ramp-down
    reg  [3:0]             slot_pos    [0:2]; // 0..15 position within current phase
    reg  [NW-1:0]          slot_coeff  [0:2]; // which coefficient index this slot targets
    reg  signed [ACC_W-1:0] slot_acc   [0:2]; //which accumulator its using

    reg  signed [DW-1:0]   coeff_mem   [0:N_COEF-1]; // 14-entry "sliding buffer"

    genvar gi;
    generate
        for (gi = 0; gi < N_COEF; gi = gi + 1) begin : FLATTEN_COEF
            assign coeff_out_flat[(gi+1)*DW-1 -: DW] = coeff_mem[gi];
        end
    endgenerate

    // ---------------------------------------------------------------
    // Combinational per-slot next-state computation
    // ---------------------------------------------------------------
    wire                     is_start     [0:2];
    wire [1:0]               eff_phase    [0:2];
    wire [3:0]               eff_pos      [0:2];
    wire                     eff_active   [0:2];
    wire [3:0]               w_sel        [0:2];
    wire signed [ACC_W-1:0]  contrib      [0:2];
    wire signed [ACC_W-1:0]  new_acc      [0:2];
    wire                     slot_last    [0:2]; 
    wire                     slot_finish  [0:2]; 

    generate
        for (gi = 0; gi < 3; gi = gi + 1) begin : SLOT_NEXT
            assign is_start[gi]   = start_pulse && (slot_of_coeff(next_coeff) == gi[1:0]);
            assign eff_phase[gi]  = is_start[gi] ? 2'd0 : slot_phase[gi];
            assign eff_pos[gi]    = is_start[gi] ? 4'd0 : slot_pos[gi];
            assign eff_active[gi] = is_start[gi] || slot_active[gi];
            assign w_sel[gi]      = weight_of(eff_phase[gi], eff_pos[gi]);
            assign contrib[gi]    = wmul(sample_in, w_sel[gi]);
            assign new_acc[gi]    = is_start[gi] ? contrib[gi] : (slot_acc[gi] + contrib[gi]);
            assign slot_last[gi]  = (eff_pos[gi] == 4'd15);
            assign slot_finish[gi]= slot_last[gi] && (eff_phase[gi] == 2'd2);
        end
    endgenerate

    integer i;

    // ---------------------------------------------------------------
    // Single sequential process: registers all state. Centralizing the
    // coeff_mem write here (instead of one write per slot's always block)
    // avoids a multiple-driver situation, since only one slot can finish
    // in any given cycle (completions are 16 cycles apart, same as starts).
    // ---------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sample_counter <= {CW{1'b0}};
            next_coeff     <= {NW{1'b0}};
            features_valid <= 1'b0;
            for (i = 0; i < 3; i = i + 1) begin
                slot_active[i] <= 1'b0;
                slot_phase[i]  <= 2'd0;
                slot_pos[i]    <= 4'd0;
                slot_coeff[i]  <= {NW{1'b0}};
                slot_acc[i]    <= {ACC_W{1'b0}};
            end
            for (i = 0; i < N_COEF; i = i + 1) coeff_mem[i] <= {DW{1'b0}};
        end else begin
            features_valid <= 1'b0;

            if (sample_valid) begin
                for (i = 0; i < 3; i = i + 1) begin
                    if (eff_active[i]) begin
                        if (slot_last[i]) begin
                            if (eff_phase[i] == 2'd2) begin
                                slot_active[i] <= 1'b0;
                                slot_phase[i]  <= 2'd0;
                                slot_pos[i]    <= 4'd0;
                            end else begin
                                slot_phase[i] <= eff_phase[i] + 2'd1;
                                slot_pos[i]   <= 4'd0;
                                slot_acc[i]   <= new_acc[i];
                                slot_active[i]<= 1'b1;
                            end
                        end else begin
                            slot_phase[i] <= eff_phase[i];
                            slot_pos[i]   <= eff_pos[i] + 4'd1;
                            slot_acc[i]   <= new_acc[i];
                            slot_active[i]<= 1'b1;
                        end
                        if (is_start[i]) slot_coeff[i] <= next_coeff;
                    end
                end

                for (i = 0; i < 3; i = i + 1) begin
                    if (slot_finish[i]) coeff_mem[slot_coeff[i]] <= sat_dw(new_acc[i] >>> 4);
                end

                if (start_pulse) next_coeff <= next_coeff + 1'b1;

                if (sample_counter == DRIVE_CYCLES-1) begin
                    sample_counter <= {CW{1'b0}};
                    next_coeff     <= {NW{1'b0}};
                    features_valid <= 1'b1;
                end else begin
                    sample_counter <= sample_counter + 1'b1;
                end
            end
        end
    end

endmodule

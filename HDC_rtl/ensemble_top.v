// ensemble_top.v
//
// Top-level controller for the HDC CHF classifier.
//
// Flow per classification (once per 1-second ECG window):
//   1. Quantize all 14 CA4 features into level indices.
//   2. Run hdc_core once, capture its decision bit as the final result.
//
// NOTE: this used to run a 3-member time-multiplexed ensemble with
// majority voting (member IDs 0/1/2). That no longer matches the design:
// hdc_params.vh sets `N_MEMBERS=1` (a single HDCClassifier is trained in
// chf_hdc_implementation.py / generate_rom.py, not 3 separate members),
// and the ROMs in basis_and_class_rom.v are only sized/loaded for member
// 0. Driving member_id=1 or 2 read past the valid ROM contents for a
// single-member build and produced garbage decisions. Simplified to a
// single pass through hdc_core, member 0 only.

`include "hdc_params.vh"

module ensemble_top (
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          start,                 
    input  wire signed [`THRESH_W-1:0]   feat_in [0:`N_FEAT-1], // 14 CA4 coefficients, Q6.10

    output reg                           result_valid,
    output reg                           result_chf             // 1 = CHF detected, 0 = normal
);

    // ---------------- Quantization stage ----------------
    reg  [3:0] quant_feat_sel;

    reg  [4:0] level_idx_reg [0:`N_FEAT-1];

    // ROM ports (single consumer at a time - quantizer's threshold
    // lookup, then the core's pos/level/class lookups; arbitrated by FSM)
    reg  [1:0] rom_pos_member;
    reg  [3:0] rom_pos_feat;
    reg  [9:0] rom_pos_bit_idx;
    wire       rom_pos_bit_out;

    reg  [1:0] rom_lvl_member;
    reg  [4:0] rom_lvl_level;
    reg  [9:0] rom_lvl_bit_idx;
    wire       rom_lvl_bit_out;

    reg  [1:0] rom_cls_member;
    reg        rom_cls_class;
    reg  [9:0] rom_cls_bit_idx;
    wire       rom_cls_bit_out;
    wire signed [7:0] rom_cls_value_out;

    reg  [3:0]  thresh_feat;
    reg  [4:0]  thresh_idx_w;
    wire signed [`THRESH_W-1:0] thresh_out;

    wire       rom_pos_rd_en, rom_lvl_rd_en, rom_cls_rd_en, rom_thresh_rd_en;

    basis_and_class_rom u_rom (
        .clk(clk),
        .pos_member(rom_pos_member), .pos_feat(rom_pos_feat),
        .pos_bit_idx(rom_pos_bit_idx), .pos_bit_out(rom_pos_bit_out),
        .lvl_member(rom_lvl_member), .lvl_level(rom_lvl_level),
        .lvl_bit_idx(rom_lvl_bit_idx), .lvl_bit_out(rom_lvl_bit_out),
        .cls_member(rom_cls_member), .cls_class(rom_cls_class),
        .cls_bit_idx(rom_cls_bit_idx), .cls_bit_out(rom_cls_bit_out),
        .cls_value_out(rom_cls_value_out),
        .thresh_feat(thresh_feat), .thresh_idx(thresh_idx_w), .thresh_out(thresh_out),
        .pos_rd_en(rom_pos_rd_en), .lvl_rd_en(rom_lvl_rd_en),
        .cls_rd_en(rom_cls_rd_en), .thresh_rd_en(rom_thresh_rd_en)
    );

    wire [4:0] lf_thresh_idx;
    wire       lf_done;
    wire [4:0] lf_level_idx;
    reg        lf_start;
    reg  signed [`THRESH_W-1:0] lf_feat_value;

    level_finder u_level_finder (
        .clk(clk), .rst_n(rst_n), .start(lf_start),
        .feat_value(lf_feat_value),
        .thresh_in(thresh_out),
        .thresh_idx(lf_thresh_idx),
        .done(lf_done),
        .level_idx(lf_level_idx),
        .thresh_rd_en(rom_thresh_rd_en)
    );
    always @(*) begin
        thresh_feat  = quant_feat_sel;
        thresh_idx_w = lf_thresh_idx;
    end

    // ---------------- Core (single pass, member 0) ----------------
    reg  [1:0] core_member_id;
    reg        core_start;
    wire       core_done;
    wire       core_decision;

    hdc_core u_core (
        .clk(clk), .rst_n(rst_n), .start(core_start), .member_id(core_member_id),
        .level_idx(level_idx_reg),
        .rom_pos_member(rom_pos_member), .rom_pos_feat(rom_pos_feat),
        .rom_pos_bit_idx(rom_pos_bit_idx), .rom_pos_bit_out(rom_pos_bit_out),
        .rom_lvl_member(rom_lvl_member), .rom_lvl_level(rom_lvl_level),
        .rom_lvl_bit_idx(rom_lvl_bit_idx), .rom_lvl_bit_out(rom_lvl_bit_out),
        .rom_cls_member(rom_cls_member), .rom_cls_class(rom_cls_class),
        .rom_cls_bit_idx(rom_cls_bit_idx), .rom_cls_value_out(rom_cls_value_out),
        .rom_pos_rd_en(rom_pos_rd_en), .rom_lvl_rd_en(rom_lvl_rd_en),
        .rom_cls_rd_en(rom_cls_rd_en),
        .done(core_done), .member_decision(core_decision)
    );

    // ---------------- Top-level FSM ----------------
    localparam S_IDLE       = 3'd0,
               S_QUANT_START= 3'd1,
               S_QUANT_WAIT = 3'd2,
               S_QUANT_NEXT = 3'd3,
               S_CORE       = 3'd4,
               S_DONE       = 3'd5;

    reg [2:0] top_state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            top_state <= S_IDLE;
            result_valid <= 1'b0;
            quant_feat_sel <= 4'd0;
            lf_start <= 1'b0;
            core_start <= 1'b0;
            core_member_id <= 2'd0;
        end else begin
            case (top_state)
                S_IDLE: begin
                    result_valid <= 1'b0;
                    if (start) begin
                        quant_feat_sel <= 4'd0;
                        top_state <= S_QUANT_START;
                    end
                end

                S_QUANT_START: begin
                    lf_feat_value <= feat_in[quant_feat_sel];
                    lf_start <= 1'b1;
                    top_state <= S_QUANT_WAIT;
                end

                S_QUANT_WAIT: begin
                    lf_start <= 1'b0;
                    if (lf_done) begin
                        level_idx_reg[quant_feat_sel] <= lf_level_idx;
                        top_state <= S_QUANT_NEXT;
                    end
                end

                S_QUANT_NEXT: begin
                    if (quant_feat_sel == `N_FEAT - 1) begin
                        core_member_id <= 2'd0;   // only member: 0
                        core_start <= 1'b1;
                        top_state <= S_CORE;
                    end else begin
                        quant_feat_sel <= quant_feat_sel + 1'b1;
                        top_state <= S_QUANT_START;
                    end
                end

                S_CORE: begin
                    core_start <= 1'b0;
                    if (core_done) begin
                        result_chf   <= core_decision;
                        top_state    <= S_DONE;
                    end
                end

                S_DONE: begin
                    result_valid <= 1'b1;
                    top_state    <= S_IDLE;
                end
            endcase
        end
    end

endmodule
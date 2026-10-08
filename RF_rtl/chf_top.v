// =============================================================================
// chf_top.v
// =============================================================================

`timescale 1ns/1ps

module chf_top #(
    parameter integer DW        = 16,  
    parameter integer ACC_W     = 26,
    parameter integer N_COEF    = 14,
    parameter integer NUM_TREES = 3
) (
    input  wire                 clk,
    input  wire                 rst_n,

    input  wire                 sample_valid,
    input  wire signed [DW-1:0] sample_in,     

    output wire                 classification_valid,
    output wire                 chf_detected,
    output wire [N_COEF*DW-1:0] feature_vector_flat
);

    wire features_valid;
    wire [N_COEF*DW-1:0] coeff_out_flat;

    wavelet_ca4 #(
        .DW(DW), .ACC_W(ACC_W), .N_COEF(N_COEF)
    ) u_wavelet (
        .clk(clk), .rst_n(rst_n),
        .sample_valid(sample_valid), .sample_in(sample_in),
        .features_valid(features_valid),
        .coeff_out_flat(coeff_out_flat)
    );

    reg [N_COEF*DW-1:0] feat_reg;
    reg                 class_valid_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            feat_reg        <= {(N_COEF*DW){1'b0}};
            class_valid_reg <= 1'b0;
        end else begin
            class_valid_reg <= features_valid;
            if (features_valid) feat_reg <= coeff_out_flat;
        end
    end

    assign feature_vector_flat = feat_reg;
    assign classification_valid = class_valid_reg;

    rf_classifier #(
        .DW(DW), .N_COEF(N_COEF), .NUM_TREES(NUM_TREES)
    ) u_rf (
        .feat_flat(feat_reg),
        .chf_detected(chf_detected)
    );

endmodule
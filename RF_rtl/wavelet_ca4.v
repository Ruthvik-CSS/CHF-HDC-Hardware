// =============================================================================
// wavelet_ca4.v
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

    reg                    slot_active [0:2];
    reg  [1:0]             slot_phase  [0:2];
    reg  [3:0]             slot_pos    [0:2];
    reg  [NW-1:0]          slot_coeff  [0:2];
    reg  signed [ACC_W-1:0] slot_acc   [0:2];

    reg  signed [DW-1:0]   coeff_mem   [0:N_COEF-1];

    genvar gi;
    generate
        for (gi = 0; gi < N_COEF; gi = gi + 1) begin : FLATTEN_COEF
            assign coeff_out_flat[(gi+1)*DW-1 -: DW] = coeff_mem[gi];
        end
    endgenerate

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
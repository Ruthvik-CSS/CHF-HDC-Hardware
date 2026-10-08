// basis_and_class_rom.v

`include "hdc_params.vh"

module basis_and_class_rom (
    input  wire clk,

    // Position ROM (1-bit)
    input  wire [1:0]  pos_member,
    input  wire [3:0]  pos_feat,
    input  wire [9:0]  pos_bit_idx,
    output reg         pos_bit_out,

    // Level ROM (1-bit)
    input  wire [1:0]  lvl_member,
    input  wire [4:0]  lvl_level,
    input  wire [9:0]  lvl_bit_idx,
    output reg         lvl_bit_out,

    // Class ROM (8-bit values)
    input  wire [1:0]  cls_member,
    input  wire        cls_class,
    input  wire [9:0]  cls_bit_idx,
    output reg         cls_bit_out,
    output reg  signed [7:0] cls_value_out,

    // Threshold ROM 
    input  wire [3:0]  thresh_feat,
    input  wire [4:0]  thresh_idx,
    output reg  signed [`THRESH_W-1:0] thresh_out,

    // read-enable ports
    input  wire         pos_rd_en,
    input  wire         lvl_rd_en,
    input  wire         cls_rd_en,
    input  wire         thresh_rd_en
);

    // Position ROM 
    reg pos_mem [0 : (`N_MEMBERS * `N_FEAT * `DIM) - 1];
    initial begin
        $readmemb("roms_500/pos_rom_m0.mem", pos_mem, 0*(`N_FEAT*`DIM), 1*(`N_FEAT*`DIM)-1);
    end
    wire [31:0] pos_addr = pos_member*(`N_FEAT*`DIM) + pos_feat*`DIM + pos_bit_idx;

    always @(posedge clk) if (pos_rd_en) pos_bit_out <= pos_mem[pos_addr];

    // Level ROM 
    reg lvl_mem [0 : (`N_MEMBERS * `N_LEVELS * `DIM) - 1];
    initial begin
        $readmemb("roms_500/level_rom_m0.mem", lvl_mem, 0*(`N_LEVELS*`DIM), 1*(`N_LEVELS*`DIM)-1);
    end
    wire [31:0] lvl_addr = lvl_member*(`N_LEVELS*`DIM) + lvl_level*`DIM + lvl_bit_idx;
    
    always @(posedge clk) if (lvl_rd_en) lvl_bit_out <= lvl_mem[lvl_addr];

    // Class ROM:
    reg signed [7:0] cls_mem [0 : (`N_MEMBERS * 2 * `DIM) - 1];
    initial begin
        $readmemh("roms_500/class_rom_m0.mem", cls_mem, 0*(2*`DIM), 1*(2*`DIM)-1);
    end
    wire [31:0] cls_addr = cls_member*(2*`DIM) + cls_class*`DIM + cls_bit_idx;
    always @(posedge clk) begin 
        if (cls_rd_en) begin
            cls_value_out <= cls_mem[cls_addr];
            cls_bit_out <= cls_mem[cls_addr][7];
        end
    end
    
    // Threshold ROM 
    reg signed [`THRESH_W-1:0] thresh_mem [0 : (`N_FEAT * (`N_LEVELS-1)) - 1];
    genvar gf;
    generate
        for (gf = 0; gf < `N_FEAT; gf = gf + 1) begin : THRESH_INIT
            initial begin
                $readmemh($sformatf("roms_500/thresh_feat%02d.mem", gf), thresh_mem,
                          gf*(`N_LEVELS-1), (gf+1)*(`N_LEVELS-1)-1);
            end
        end
    endgenerate
    wire [31:0] thresh_addr = thresh_feat*(`N_LEVELS-1) + thresh_idx;
    always @(posedge clk) if (thresh_rd_en) thresh_out <= thresh_mem[thresh_addr];

endmodule
"""Schema definitions for the two processed lifetime-data export formats."""

SCHEMA_TYPE1 = dict(
    ccm_filename="CCM.pkl.gz",
    cd_filename="CD.pkl.gz",
    ccm_time_mode="ms_col",
    ccm_time_col="Time [ms]",
    ccm_test_type_col="Test type",
    ccm_test_type_value="CYC",
    ccm_charge_ind="charge_cycle_indicator",
    ccm_discharge_ind="discharge_cycle_indicator",
    ccm_ahthr_col="Ah throughput [A.h]",
    ccm_test_name_col="Test name",
    ccm_rpt_type_value="RPT",
    ccm_protocol_col="Protocol",
    cd_time_mode="ms_col",
    cd_time_col="Time [ms]",
    cd_v_col="Voltage [V]",
    cd_i_col="Current [A]",
    cd_ah_col="Ah throughput [A.h]",
)

SCHEMA_TYPE2 = dict(
    ccm_filename="cell_cycle_metrics.pkl.gz",
    cd_filename="cell_data.pkl.gz",
    ccm_time_mode="timestamp",
    ccm_time_col="timestamp",
    ccm_test_type_col="cycle type",
    ccm_test_type_value="CYC",
    ccm_charge_ind="charge cycle indicator",
    ccm_discharge_ind="discharge cycle indicator",
    ccm_ahthr_col="capacity(ah)",
    ccm_test_name_col="test name",
    ccm_cycle_index_col="cycle index",
    ccm_rpt_type_value="RPT",
    ccm_protocol_col="protocol",
    cd_time_mode="timestamp",
    cd_time_col="timestamp",
    cd_v_col="voltage(v)",
    cd_i_col="current(a)",
    cd_ah_col="capacity(ah)",
)

TYPE1_CELLS = [1, 3, 4, 5, 8, 9, 18, 21, 22, 23, 24, 26, 28, 30, 31, 35, 36, 40, 41, 42, 43, 44, 45, 46, 48,
               51, 52, 53, 54, 57, 58, 59, 61, 65, 66, 67, 68, 69, 73, 75]
TYPE2_CELLS = [2, 7, 11, 12, 16, 17, 19, 20, 25, 27, 32, 47, 49, 55, 56, 60, 64, 71, 72, 74]
PREFER_OVERLAP = "type2"


def schema_id_for_cell(cell: int):
    in1 = cell in set(TYPE1_CELLS)
    in2 = cell in set(TYPE2_CELLS)
    if in1 and not in2:
        return "type1"
    if in2 and not in1:
        return "type2"
    if in1 and in2:
        return PREFER_OVERLAP
    return None

#!/bin/bash

module load cdo

# this function saves LMDz run-nudg-6h-1 files for the periods 1985-1999 and 2000-2014 from the files of the period 1985-2014 :
separate_periods_lmdz_data() {
    dir_name=/lustre/fsstor/projects/rech/huw/uzb76ep/LMDZ/run-nudg-6h-1/bilinear
    # start end period 1 :
    start_1=1985
    end_1=1999
    start_2=2000
    end_2=2014
    var="$1"

    file_input=${dir_name}/${var}_france_6h_1985_2014_sorted_regrid_bilinear.nc
    file_output_1=${dir_name}/${var}_france_6h_${start_1}_${end_1}_sorted_regrid_bilinear.nc
    file_output_2=${dir_name}/${var}_france_6h_${start_2}_${end_2}_sorted_regrid_bilinear.nc

    cdo selyear,${start_1}/${end_1} ${file_input} ${file_output_1}
    cdo selyear,${start_2}/${end_2} ${file_input} ${file_output_2}
    echo ${var} done
}

separate_periods_lmdz_data u850
separate_periods_lmdz_data v850
separate_periods_lmdz_data t850
separate_periods_lmdz_data prw
separate_periods_lmdz_data u700
separate_periods_lmdz_data u500
separate_periods_lmdz_data v700
separate_periods_lmdz_data v500
separate_periods_lmdz_data t700
separate_periods_lmdz_data t500
separate_periods_lmdz_data q850
separate_periods_lmdz_data q700
separate_periods_lmdz_data q500
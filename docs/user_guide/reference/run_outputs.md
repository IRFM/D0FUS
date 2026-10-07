(chap:run_outputs)=

# Run mode outputs

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix D.11. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


This page lists the 167 scalar outputs of the Run mode, organised in 13 families. The list is generated automatically from a converged run of the ITER reference deck.

| p9.2cmp4.4cm@                                      | Unit         |     |
|:---------------------------------------------------|:-------------|:----|
| **Plasma geometry & shape**                        |              |     |
| Major radius R0                                    | m            |     |
| Minor radius a                                     | m            |     |
| Aspect ratio A = R0/a                              | \-           |     |
| Elongation kappa_edge (LCFS)                       | \-           |     |
| Elongation kappa_95                                | \-           |     |
| Triangularity delta_edge (LCFS)                    | \-           |     |
| Triangularity delta_95                             | \-           |     |
| Plasma volume                                      | m^3          |     |
| Plasma surface                                     | m^2          |     |
| Minor radius (LCFS) r_minor                        | m            |     |
| Separatrix radius r_sep                            | m            |     |
| r_c                                                | m            |     |
| r_d                                                | m            |     |
| **Magnetic field & current**                       |              |     |
| Toroidal field on axis B0                          | T            |     |
| Peak field B_max                                   | T            |     |
| CS field B_CS                                      | T            |     |
| Poloidal field B_pol                               | T            |     |
| Plasma current Ip                                  | MA           |     |
| Bootstrap current Ib                               | A            |     |
| Current-drive current I_CD                         | A            |     |
| Ohmic current I_Ohm                                | A            |     |
| Edge safety factor q95                             | \-           |     |
| Cylindrical safety factor q\*                      | \-           |     |
| Loop voltage Vloop                                 | V            |     |
| Internal inductance li                             | \-           |     |
| **Density, temperature, pressure, beta**           |              |     |
| Volume-avg density nbar                            | m^-3         |     |
| Line-avg density nbar_line                         | m^-3         |     |
| Greenwald density nG                               | m^-3         |     |
| Volume-avg temperature Tbar                        | keV          |     |
| Volume-avg pressure pbar                           | Pa           |     |
| Effective charge Z_eff                             | \-           |     |
| Helium fraction f_He                               | \-           |     |
| Normalised beta betaN                              | \-           |     |
| Toroidal beta betaT                                | \-           |     |
| Poloidal beta betaP                                | \-           |     |
| Total normalised beta betaN_total                  | \-           |     |
| Fast-alpha beta beta_fast_alpha                    | \-           |     |
| Density peaking nu_n                               | \-           |     |
| Temperature peaking nu_T                           | \-           |     |
| Pedestal radius rho_ped                            | \-           |     |
| Pedestal density fraction n_ped_frac               | \-           |     |
| Pedestal temperature fraction T_ped_frac           | \-           |     |
| **Confinement & fusion**                           |              |     |
| Energy confinement time tauE                       | s            |     |
| Thermal stored energy W_th                         | J            |     |
| Fusion power P_fus                                 | MW           |     |
| Fusion gain Q                                      | \-           |     |
| Alpha heating fraction f_alpha                     | \-           |     |
| Alpha slowing-down time tau_alpha                  | s            |     |
| Fast-alpha slowing time tau_sd_alpha               | s            |     |
| Fast-alpha stored energy W_fast_alpha              | J            |     |
| **Power balance**                                  |              |     |
| Auxiliary heating P_aux                            | MW           |     |
| Current-drive power P_CD                           | MW           |     |
| Power across separatrix P_sep                      | MW           |     |
| L-H threshold power P_Thresh                       | MW           |     |
| Net electric power P_elec                          | MWe          |     |
| Wall-plug power P_wallplug                         | MW           |     |
| CD efficiency eta_CD                               | A/W          |     |
| Bremsstrahlung P_Brem                              | MW           |     |
| Synchrotron P_syn                                  | MW           |     |
| Line radiation P_line                              | MW           |     |
| Core line radiation P_line_core                    | MW           |     |
| **Current-drive breakdown**                        |              |     |
| LH efficiency eta_LH                               | \-           |     |
| EC efficiency eta_EC                               | \-           |     |
| NBI efficiency eta_NBI                             | \-           |     |
| LH power P_LH                                      | MW           |     |
| EC power P_EC                                      | MW           |     |
| NBI power P_NBI                                    | MW           |     |
| ICRH power P_ICR                                   | MW           |     |
| LH-driven current I_LH                             | A            |     |
| EC-driven current I_EC                             | A            |     |
| NBI-driven current I_NBI                           | A            |     |
| **Poloidal flux budget**                           |              |     |
| Plasma initiation flux psi_PI                      | Wb           |     |
| Ramp-up flux psi_RampUp                            | Wb           |     |
| Plateau flux psi_plateau                           | Wb           |     |
| PF flux psi_PF                                     | Wb           |     |
| CS flux psi_CS                                     | Wb           |     |
| **Divertor & exhaust**                             |              |     |
| SOL elongation factor f_kappa_SOL                  | \-           |     |
| Divertor heat load (total) heat                    | MW/m^2       |     |
| Parallel heat flux heat_par                        | MW/m^2       |     |
| Poloidal heat flux heat_pol                        | MW/m^2       |     |
| SOL power width lambda_q                           | m            |     |
| Target heat flux q_target                          | MW/m^2       |     |
| Wall radiated power P_wall_rad                     | MW           |     |
| Divertor radiated power P_wall_div                 | MW           |     |
| Neutron wall load Gamma_n                          | MW/m^2       |     |
| **TF coil (winding pack, composition, structure)** |              |     |
| TF current density J_TF                            | A/m^2        |     |
| TF radial thickness c_TF                           | m            |     |
| TF winding-pack thickness c_WP                     | m            |     |
| TF nose thickness c_nose                           | m            |     |
| TF shape exponent n_shape_TF                       | \-           |     |
| Number of TF coils N_TF                            | \-           |     |
| TF steel fraction Steel_fraction_TF                | \-           |     |
| TF axial stress sz_TF                              | Pa           |     |
| TF hoop stress st_TF                               | Pa           |     |
| TF radial stress sr_TF                             | Pa           |     |
| TF von Mises stress sf_TF                          | Pa           |     |
| TF SC fraction f_sc_TF                             | \-           |     |
| TF Cu fraction f_cu_TF                             | \-           |     |
| TF He-pipe fraction f_He_pipe_TF                   | \-           |     |
| TF void fraction f_void_TF                         | \-           |     |
| TF He fraction f_He_TF                             | \-           |     |
| TF insulator fraction f_In_TF                      | \-           |     |
| TF steel volume V_steel_TF                         | m^3          |     |
| TF SC volume V_sc_TF                               | m^3          |     |
| TF Cu volume V_cu_TF                               | m^3          |     |
| TF He volume V_He_TF                               | m^3          |     |
| TF insulator volume V_In_TF                        | m^3          |     |
| Single-TF volume V_TF_one                          | m^3          |     |
| TF cable length L_cable_TF                         | m            |     |
| TF strand count n_sc_TF                            | \-           |     |
| TF strand length L_sc_strand_TF                    | m            |     |
| TF steel mass M_steel_TF                           | kg           |     |
| TF SC mass M_sc_TF                                 | kg           |     |
| TF Cu mass M_cu_TF                                 | kg           |     |
| TF insulator mass M_In_TF                          | kg           |     |
| TF total mass M_total_TF                           | kg           |     |
| **CS coil (winding pack, composition, structure)** |              |     |
| CS current density J_CS_1                          | A/m^2        |     |
| CS field (alt.) B_CS2                              | T            |     |
| CS current density (alt.) J_CS2                    | A/m^2        |     |
| CS radial thickness c_CS                           | m            |     |
| CS module count N_sub_CS                           | \-           |     |
| CS shape exponent n_shape_CS                       | \-           |     |
| CS steel fraction Steel_fraction_CS                | \-           |     |
| CS axial stress sz_CS                              | Pa           |     |
| CS hoop stress st_CS                               | Pa           |     |
| CS radial stress sr_CS                             | Pa           |     |
| CS von Mises stress sf_CS                          | Pa           |     |
| CS SC fraction f_sc_CS                             | \-           |     |
| CS Cu fraction f_cu_CS                             | \-           |     |
| CS He-pipe fraction f_He_pipe_CS                   | \-           |     |
| CS void fraction f_void_CS                         | \-           |     |
| CS He fraction f_He_CS                             | \-           |     |
| CS insulator fraction f_In_CS                      | \-           |     |
| CS steel volume V_steel_CS                         | m^3          |     |
| CS SC volume V_sc_CS                               | m^3          |     |
| CS Cu volume V_cu_CS                               | m^3          |     |
| CS He volume V_He_CS                               | m^3          |     |
| CS insulator volume V_In_CS                        | m^3          |     |
| CS solenoid volume V_CS_geom                       | m^3          |     |
| CS cable length L_cable_CS                         | m            |     |
| CS strand count n_sc_CS                            | \-           |     |
| CS strand length L_sc_strand_CS                    | m            |     |
| CS steel mass M_steel_CS                           | kg           |     |
| CS SC mass M_sc_CS                                 | kg           |     |
| CS Cu mass M_cu_CS                                 | kg           |     |
| CS insulator mass M_In_CS                          | kg           |     |
| CS total mass M_total_CS                           | kg           |     |
| **Radial build volumes & masses**                  |              |     |
| Blanket+shield radial thickness b                  | m            |     |
| Outboard clearance Delta_TF                        | m            |     |
| Assembly gap Gap                                   | m            |     |
| First-wall thickness e_fw                          | m            |     |
| Blanket thickness e_blanket                        | m            |     |
| Shield thickness e_shield                          | m            |     |
| Blanket torus volume V_blanket                     | m^3          |     |
| First-wall mass M_rb_FW                            | kg           |     |
| Breeding-blanket mass M_rb_BB                      | kg           |     |
| Shield mass M_rb_shield                            | kg           |     |
| Vacuum-vessel mass M_rb_VV                         | kg           |     |
| Divertor mass M_rb_divertor                        | kg           |     |
| Radial-build total mass M_rb_total                 | kg           |     |
| **Component lifetime & availability**              |              |     |
| Blanket calendar lifetime t_life_bl_yr             | yr           |     |
| Divertor calendar lifetime t_life_div_yr           | yr           |     |
| Operation time per cycle T_op_limit                | yr           |     |
| Capacity factor CF                                 | \-           |     |
| **Techno-economics (Sheffield 2016)**              |              |     |
| Geometric cost proxy (V_build / P_fus)             | m^3/MW       |     |
| Cost of electricity COE                            | EUR/MWh      |     |
| Total constructed capital C_invest                 | M EUR (2025) |     |

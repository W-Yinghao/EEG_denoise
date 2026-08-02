# SGEYESUB corrected operator audit

This audit is additive and read-only with respect to earlier result files. Blocked, ineligible, failed, and fallback rows remain in coverage denominators but are excluded from performance means.

Audit status: `complete_43_success_paired`.

Frozen development gamma: `0`.

Scientific interpretation: `hard_Q_P0_tradeoff_inconclusive`.
This is a post-hoc descriptive audit, is non-preregistered, and is not formal gate evidence. Matching P0 showed descriptively lower held-out EOG remaining ratios and higher coherence reduction than population. Non-artifact preservation and covariance/PSD distortion were roughly tied (their descriptive paired confidence intervals spanned zero), while the ERP preservation proxy was slightly lower; the absolute hard-Q safety thresholds were not met. No broad category-level failure decision is generated.
Required focus/control coverage: `complete_44_unique_recording_keys`.

Interpretation: `development_selected_population_endpoint`. gamma=0 sets the full and both split-half shrinkage projectors to the same population projector, so its stability component is structurally zero; this is an endpoint property, not evidence that participant calibration is stable. The support-only objective is a conservative selection rule, not an unbiased hypothesis test of personalization. Evaluation continues; this endpoint is not itself a negative held-out personalization result.

## Development gamma score components

| Gamma | Success records | Mean stability | Mean capture loss | Weighted capture | Mean score |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 15 | 0.0 | 0.026146352050426678 | 0.013073176025213339 | 0.013073176025213339 |
| 0.25 | 15 | 0.024733298168717714 | 0.01495131755167789 | 0.007475658775838945 | 0.03220895694455666 |
| 0.5 | 15 | 0.050301967262005 | 0.006623176795064941 | 0.0033115883975324703 | 0.05361355565953748 |
| 0.75 | 15 | 0.07564022374354527 | 0.001617066405322301 | 0.0008085332026611505 | 0.07644875694620643 |
| 1.0 | 15 | 0.09971933807191732 | 1.792814785406365e-31 | 8.964073927031825e-32 | 0.09971933807191732 |

## Required methods: nine metrics and status coverage

Status columns are success/fallback/blocked/ineligible/failed over all registered records; numeric summaries use only successful, non-fallback finite rows.

| Method | Metric | Direction | Finite N | Mean | Median | S/F/B/I/X |
|---|---|---|---:|---:|---:|---:|
| matching_Qy | matching_projector_attenuation_db | higher | 44 | 310.38565966844936 | 309.6510972760187 | 44/0/0/0/0 |
| matching_Qy | population_projector_attenuation_db | higher | 43 | 30.87032418968814 | 30.551205762461194 | 44/0/0/0/0 |
| matching_Qy | nonartifact_observation_preservation | higher | 44 | 0.27150470641762187 | 0.27594961846107613 | 44/0/0/0/0 |
| matching_Qy | eog_coherence_reduction | higher | 44 | 0.4099916360195609 | 0.402293252339345 | 44/0/0/0/0 |
| matching_Qy | reference_free_psd_distortion | lower | 44 | 0.6682485897895 | 0.6446336444688172 | 44/0/0/0/0 |
| matching_Qy | reference_free_covariance_distortion | lower | 44 | 0.9096038404609073 | 0.9093942950702072 | 44/0/0/0/0 |
| matching_Qy | heldout_eog_prediction_remaining_ratio | lower | 44 | 0.6093153054098325 | 0.5494659345472332 | 44/0/0/0/0 |
| matching_Qy | condition_erp_observation_relative_preservation | higher | 42 | 0.05546414605030129 | 0.048577372539513886 | 44/0/0/0/0 |
| matching_Qy | observation_change_ratio | lower | 44 | 0.8940394028059706 | 0.8996877866982633 | 44/0/0/0/0 |
| pop_Qy | matching_projector_attenuation_db | higher | 43 | 29.622220183391015 | 29.074870992257726 | 43/0/1/0/0 |
| pop_Qy | population_projector_attenuation_db | higher | 43 | 310.48966460016385 | 310.6772478479429 | 43/0/1/0/0 |
| pop_Qy | nonartifact_observation_preservation | higher | 43 | 0.27232699338605143 | 0.27528303238952445 | 43/0/1/0/0 |
| pop_Qy | eog_coherence_reduction | higher | 43 | 0.3268566329862904 | 0.31253014732777595 | 43/0/1/0/0 |
| pop_Qy | reference_free_psd_distortion | lower | 43 | 0.6684893581814213 | 0.6298786827889581 | 43/0/1/0/0 |
| pop_Qy | reference_free_covariance_distortion | lower | 43 | 0.9119548724565787 | 0.9119218568836772 | 43/0/1/0/0 |
| pop_Qy | heldout_eog_prediction_remaining_ratio | lower | 43 | 0.6254013309879394 | 0.5676974996954826 | 43/0/1/0/0 |
| pop_Qy | condition_erp_observation_relative_preservation | higher | 41 | 0.06099649396086298 | 0.05258785572253599 | 43/0/1/0/0 |
| pop_Qy | observation_change_ratio | lower | 43 | 0.8895336367095389 | 0.898426098543697 | 43/0/1/0/0 |
| B6_Qy__gamma_0 | matching_projector_attenuation_db | higher | 43 | 29.622220183391015 | 29.074870992257726 | 43/0/1/0/0 |
| B6_Qy__gamma_0 | population_projector_attenuation_db | higher | 43 | 310.48966460016385 | 310.6772478479429 | 43/0/1/0/0 |
| B6_Qy__gamma_0 | nonartifact_observation_preservation | higher | 43 | 0.27232699338605143 | 0.27528303238952445 | 43/0/1/0/0 |
| B6_Qy__gamma_0 | eog_coherence_reduction | higher | 43 | 0.3268566329862904 | 0.31253014732777595 | 43/0/1/0/0 |
| B6_Qy__gamma_0 | reference_free_psd_distortion | lower | 43 | 0.6684893581814213 | 0.6298786827889581 | 43/0/1/0/0 |
| B6_Qy__gamma_0 | reference_free_covariance_distortion | lower | 43 | 0.9119548724565787 | 0.9119218568836772 | 43/0/1/0/0 |
| B6_Qy__gamma_0 | heldout_eog_prediction_remaining_ratio | lower | 43 | 0.6254013309879394 | 0.5676974996954826 | 43/0/1/0/0 |
| B6_Qy__gamma_0 | condition_erp_observation_relative_preservation | higher | 41 | 0.06099649396086298 | 0.05258785572253599 | 43/0/1/0/0 |
| B6_Qy__gamma_0 | observation_change_ratio | lower | 43 | 0.8895336367095389 | 0.898426098543697 | 43/0/1/0/0 |
| B6_soft_proximal__gamma_0 | matching_projector_attenuation_db | higher | 43 | 11.634242436274628 | 11.698020799786068 | 43/0/1/0/0 |
| B6_soft_proximal__gamma_0 | population_projector_attenuation_db | higher | 43 | 12.041199826559248 | 12.04119982655925 | 43/0/1/0/0 |
| B6_soft_proximal__gamma_0 | nonartifact_observation_preservation | higher | 43 | 0.45424524503953856 | 0.45646227429214326 | 43/0/1/0/0 |
| B6_soft_proximal__gamma_0 | eog_coherence_reduction | higher | 43 | 0.17992777918986338 | 0.17815738507578505 | 43/0/1/0/0 |
| B6_soft_proximal__gamma_0 | reference_free_psd_distortion | lower | 43 | 0.5669697793933219 | 0.5284941901574245 | 43/0/1/0/0 |
| B6_soft_proximal__gamma_0 | reference_free_covariance_distortion | lower | 43 | 0.7963609220553104 | 0.7906410376868941 | 43/0/1/0/0 |
| B6_soft_proximal__gamma_0 | heldout_eog_prediction_remaining_ratio | lower | 43 | 0.5425646801694675 | 0.49180809854082685 | 43/0/1/0/0 |
| B6_soft_proximal__gamma_0 | condition_erp_observation_relative_preservation | higher | 41 | 0.29574737047064725 | 0.289440891791902 | 43/0/1/0/0 |
| B6_soft_proximal__gamma_0 | observation_change_ratio | lower | 43 | 0.6671502275321542 | 0.6738195739077726 | 43/0/1/0/0 |
| native_sgeyesub_python_release_internal | matching_projector_attenuation_db | higher | 44 | 8.604687829064849 | 8.590427223086833 | 44/0/0/0/0 |
| native_sgeyesub_python_release_internal | population_projector_attenuation_db | higher | 43 | 8.629152445992874 | 8.47357553462392 | 44/0/0/0/0 |
| native_sgeyesub_python_release_internal | nonartifact_observation_preservation | higher | 44 | 0.7992083566602328 | 0.8190897233987733 | 44/0/0/0/0 |
| native_sgeyesub_python_release_internal | eog_coherence_reduction | higher | 44 | 0.4003398564047439 | 0.3876020448797528 | 44/0/0/0/0 |
| native_sgeyesub_python_release_internal | reference_free_psd_distortion | lower | 44 | 0.1964054570244328 | 0.18441376335494958 | 44/0/0/0/0 |
| native_sgeyesub_python_release_internal | reference_free_covariance_distortion | lower | 44 | 0.07505198635230852 | 0.06017480527166673 | 44/0/0/0/0 |
| native_sgeyesub_python_release_internal | heldout_eog_prediction_remaining_ratio | lower | 44 | 0.19825847296238885 | 0.17267980074530564 | 44/0/0/0/0 |
| native_sgeyesub_python_release_internal | condition_erp_observation_relative_preservation | higher | 42 | 0.1057325855463936 | 0.11991404255357696 | 44/0/0/0/0 |
| native_sgeyesub_python_release_internal | observation_change_ratio | lower | 44 | 0.7721013588313593 | 0.7703187263294163 | 44/0/0/0/0 |

## Wrong and shuffled controls

| Method | Metric | Direction | Finite N | Mean | Median | S/F/B/I/X |
|---|---|---|---:|---:|---:|---:|
| wrong_Qy | matching_projector_attenuation_db | higher | 43 | 30.920688512422874 | 30.176064092656055 | 43/0/1/0/0 |
| wrong_Qy | population_projector_attenuation_db | higher | 43 | 34.604415073929275 | 37.195146848581814 | 43/0/1/0/0 |
| wrong_Qy | nonartifact_observation_preservation | higher | 43 | 0.2740685652246832 | 0.28847254690380086 | 43/0/1/0/0 |
| wrong_Qy | eog_coherence_reduction | higher | 43 | 0.3250874812559923 | 0.32147356545084926 | 43/0/1/0/0 |
| wrong_Qy | reference_free_psd_distortion | lower | 43 | 0.6589538804858284 | 0.6151746122984102 | 43/0/1/0/0 |
| wrong_Qy | reference_free_covariance_distortion | lower | 43 | 0.9077026418067762 | 0.9073953975527845 | 43/0/1/0/0 |
| wrong_Qy | heldout_eog_prediction_remaining_ratio | lower | 43 | 0.6231765549877066 | 0.5725724728739977 | 43/0/1/0/0 |
| wrong_Qy | condition_erp_observation_relative_preservation | higher | 41 | 0.06100913873027179 | 0.051628936948816206 | 43/0/1/0/0 |
| wrong_Qy | observation_change_ratio | lower | 43 | 0.8887328127381501 | 0.900271302442637 | 43/0/1/0/0 |
| shuffled_Qy | matching_projector_attenuation_db | higher | 0 | None | None | 0/43/0/1/0 |
| shuffled_Qy | population_projector_attenuation_db | higher | 0 | None | None | 0/43/0/1/0 |
| shuffled_Qy | nonartifact_observation_preservation | higher | 0 | None | None | 0/43/0/1/0 |
| shuffled_Qy | eog_coherence_reduction | higher | 0 | None | None | 0/43/0/1/0 |
| shuffled_Qy | reference_free_psd_distortion | lower | 0 | None | None | 0/43/0/1/0 |
| shuffled_Qy | reference_free_covariance_distortion | lower | 0 | None | None | 0/43/0/1/0 |
| shuffled_Qy | heldout_eog_prediction_remaining_ratio | lower | 0 | None | None | 0/43/0/1/0 |
| shuffled_Qy | condition_erp_observation_relative_preservation | higher | 0 | None | None | 0/43/0/1/0 |
| shuffled_Qy | observation_change_ratio | lower | 0 | None | None | 0/43/0/1/0 |

## Matching versus population

Compatible stems: `43`; successful method pairs: `43`. Positive directional improvement always favors matching P0; raw delta is always `matching − population` before direction adjustment.

| Metric | Finite pairs | Raw mean delta | Raw median delta | Raw mean 95% CI | Directional mean | Wins/Ties/Losses |
|---|---:|---:|---:|---:|---:|---:|
| matching_projector_attenuation_db | 43 | 280.8215057387681 | 280.82020939484136 | [278.474466216842, 283.08211966674475] | 280.8215057387681 | 43/0/0 |
| population_projector_attenuation_db | 43 | -279.6193404104757 | -279.54634070052464 | [-281.9968271313271, -277.2056098863312] | -279.6193404104757 | 0/0/43 |
| nonartifact_observation_preservation | 43 | 0.0013242643365696171 | 0.002729384661793288 | [-0.005682144065394272, 0.0080774784156889] | 0.0013242643365696171 | 22/0/21 |
| eog_coherence_reduction | 43 | 0.07904754253638534 | 0.06983190007445073 | [0.0571985791138961, 0.10099815894454499] | 0.07904754253638534 | 41/0/2 |
| reference_free_psd_distortion | 43 | -0.003809565791286152 | -0.0056372461512144145 | [-0.014733571816179139, 0.007498187706893455] | 0.003809565791286152 | 24/0/19 |
| reference_free_covariance_distortion | 43 | -0.003936385949804714 | -0.00212594004149258 | [-0.010558240684851979, 0.002730998665926132] | 0.003936385949804714 | 23/0/20 |
| heldout_eog_prediction_remaining_ratio | 43 | -0.014066221419117673 | -0.012717330401888938 | [-0.021330784111095773, -0.006442395091249357] | 0.014066221419117673 | 31/0/12 |
| condition_erp_observation_relative_preservation | 41 | -0.004741110180182477 | -0.00434323282995952 | [-0.007960299289600186, -0.0006720820411838899] | -0.004741110180182477 | 5/0/36 |
| observation_change_ratio | 43 | 0.0031959707809797915 | 0.002105826542359579 | [-0.0007210346452999014, 0.006656299469757745] | -0.0031959707809797915 | 17/0/26 |

## Study heterogeneity

| Study | Metric | Finite/compatible | Raw mean delta | Raw median delta | Directional mean | Wins/Ties/Losses | Raw mean 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|
| study02 | matching_projector_attenuation_db | 15/15 | 279.82086073441405 | 280.379818396889 | 279.82086073441405 | 15/0/0 | [277.0500438975936, 282.60261475853156] |
| study04 | matching_projector_attenuation_db | 15/15 | 287.1374474589455 | 287.9468253381846 | 287.1374474589455 | 15/0/0 | [284.4870799997893, 289.6976105008485] |
| study05 | matching_projector_attenuation_db | 13/13 | 274.6884710666641 | 275.75807083128984 | 274.6884710666641 | 13/0/0 | [270.8268810854714, 278.2626923946638] |
| study02 | population_projector_attenuation_db | 15/15 | -278.81902491219245 | -280.1379861770583 | -278.81902491219245 | 0/0/15 | [-281.8676896041899, -275.5878964345353] |
| study04 | population_projector_attenuation_db | 15/15 | -285.36103633663254 | -283.535072284573 | -285.36103633663254 | 0/0/15 | [-288.72887897833374, -282.14489039304914] |
| study05 | population_projector_attenuation_db | 13/13 | -273.9177476090833 | -275.9112605994407 | -273.9177476090833 | 0/0/13 | [-277.82813901938425, -269.96757576077033] |
| study02 | nonartifact_observation_preservation | 15/15 | -0.004186180115750138 | -0.0038842017673119678 | -0.004186180115750138 | 7/0/8 | [-0.019655424225264336, 0.010600257282864891] |
| study04 | nonartifact_observation_preservation | 15/15 | 0.008820195835385054 | 0.01365868417610494 | 0.008820195835385054 | 10/0/5 | [-0.00274590665226972, 0.019668769631495806] |
| study05 | nonartifact_observation_preservation | 13/13 | -0.0009666822555407859 | -0.003334478233733007 | -0.0009666822555407859 | 5/0/8 | [-0.003852003780257547, 0.0019586208716076423] |
| study02 | eog_coherence_reduction | 15/15 | 0.08350770159919864 | 0.06444055406370403 | 0.08350770159919864 | 14/0/1 | [0.05159611746514425, 0.12167927199794439] |
| study04 | eog_coherence_reduction | 15/15 | 0.09505162103948626 | 0.09414057443928037 | 0.09505162103948626 | 14/0/1 | [0.04628879428255409, 0.13884908243338895] |
| study05 | eog_coherence_reduction | 13/13 | 0.05543496072956129 | 0.041415537077371334 | 0.05543496072956129 | 13/0/0 | [0.037687303703870426, 0.07611413187997951] |
| study02 | reference_free_psd_distortion | 15/15 | 0.006206390984874462 | -0.0026514994557562677 | -0.006206390984874462 | 8/0/7 | [-0.017605099810702348, 0.031087757907770585] |
| study04 | reference_free_psd_distortion | 15/15 | -0.01859511002503055 | -0.016607574244111523 | 0.01859511002503055 | 11/0/4 | [-0.0357178573553285, -0.0009425474608153659] |
| study05 | reference_free_psd_distortion | 13/13 | 0.0016938043520797496 | 0.006022837139472936 | -0.0016938043520797496 | 5/0/8 | [-0.004946571224469114, 0.008001225747990169] |
| study02 | reference_free_covariance_distortion | 15/15 | 0.0038285763893591766 | 0.0006740393202655515 | -0.0038285763893591766 | 6/0/9 | [-0.0098564727299945, 0.017909257483487973] |
| study04 | reference_free_covariance_distortion | 15/15 | -0.01539915691622933 | -0.017148414809418422 | 0.01539915691622933 | 12/0/3 | [-0.02648213268279438, -0.005163703205350668] |
| study05 | reference_free_covariance_distortion | 13/13 | 0.0003303163124191988 | 0.0011642178247828205 | -0.0003303163124191988 | 5/0/8 | [-0.0017496691755642191, 0.0023262836733666725] |
| study02 | heldout_eog_prediction_remaining_ratio | 15/15 | -0.009044395236995765 | -0.012717330401888938 | 0.009044395236995765 | 11/0/4 | [-0.022879804857708626, 0.006700962063892226] |
| study04 | heldout_eog_prediction_remaining_ratio | 15/15 | -0.030269400521280496 | -0.028500850620876983 | 0.030269400521280496 | 14/0/1 | [-0.04069180773836328, -0.019645004107572493] |
| study05 | heldout_eog_prediction_remaining_ratio | 13/13 | -0.0011646603575320008 | 0.00030773206301271117 | 0.0011646603575320008 | 6/0/7 | [-0.004590987458857268, 0.002203190829388944] |
| study02 | condition_erp_observation_relative_preservation | 15/15 | -0.007023812571571875 | -0.005847632464396235 | -0.007023812571571875 | 0/0/15 | [-0.011040480769895122, -0.004130983765998704] |
| study04 | condition_erp_observation_relative_preservation | 15/15 | -0.004626148798824769 | -0.008409655093331914 | -0.004626148798824769 | 2/0/13 | [-0.011836584360469698, 0.0057424945773969635] |
| study05 | condition_erp_observation_relative_preservation | 11/13 | -0.0017850997119574447 | -0.0019409959346623484 | -0.0017850997119574447 | 3/0/8 | [-0.0028519288603019092, -0.0007582270292005648] |
| study02 | observation_change_ratio | 15/15 | 0.006725970407726852 | 0.0050039924889419 | -0.006725970407726852 | 6/0/9 | [0.001777109864093642, 0.012417822162656867] |
| study04 | observation_change_ratio | 15/15 | 0.0009855362348448378 | 0.0019082290535382684 | -0.0009855362348448378 | 7/0/8 | [-0.008526249971392001, 0.008682290243083701] |
| study05 | observation_change_ratio | 13/13 | 0.0016733956879658217 | 0.0024012115622689123 | -0.0016733956879658217 | 4/0/9 | [0.00036516773581550896, 0.00291393803203162] |

## Hard-Q absolute safety

| Method | Finite rows | Mean/median preservation | Mean/median covariance distortion | Joint passes | Pass fraction finite/all |
|---|---:|---:|---:|---:|---:|
| matching_Qy | 44 | 0.27150470641762187/0.27594961846107613 | 0.9096038404609073/0.9093942950702072 | 0 | 0.0/0.0 |
| pop_Qy | 43 | 0.27232699338605143/0.27528303238952445 | 0.9119548724565787/0.9119218568836772 | 0 | 0.0/0.0 |
| B6_Qy__gamma_0 | 43 | 0.27232699338605143/0.27528303238952445 | 0.9119548724565787/0.9119218568836772 | 0 | 0.0/0.0 |
| wrong_Qy | 43 | 0.2740685652246832/0.28847254690380086 | 0.9077026418067762/0.9073953975527845 | 0 | 0.0/0.0 |
| shuffled_Qy | 0 | None/None | None/None | 0 | None/0.0 |

## Additional focus-method safety

| Method | Finite rows | Mean/median preservation | Mean/median covariance distortion | Joint passes | Pass fraction finite/all |
|---|---:|---:|---:|---:|---:|
| B6_soft_proximal__gamma_0 | 43 | 0.45424524503953856/0.45646227429214326 | 0.7963609220553104/0.7906410376868941 | 0 | 0.0/0.0 |
| native_sgeyesub_python_release_internal | 44 | 0.7992083566602328/0.8190897233987733 | 0.07505198635230852/0.06017480527166673 | 3 | 0.06818181818181818/0.06818181818181818 |

The study heterogeneity table contains `27` metric-study rows. The CSV safety table retains all methods; the report separately shows hard-Q and additional focus methods. No clean-target claim is made.

The two historical `e_parallel` denominator conventions, their units, and the
result families that used each convention are documented side by side in
`reports/e_parallel_formula_side_by_side.md`. SGEYESUB has no paired clean
target, so this natural-EEG audit does not compute or reinterpret
`e_parallel`.

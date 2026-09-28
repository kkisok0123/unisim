# Apple-stem contact-force measurements

Measured 2026-09-28 on the current task assets and controller schedule. Each run
covered 0–14 s. The tables sample once every 0.05 s and report **force on the
apple** from each right-hand pad in world coordinates, in newtons. `Mean |F|` is
the mean of the per-sample force magnitudes, rather than the magnitude of the
mean vector. The lift window has 20 samples (8.00–8.95 s); the hold window has
60 samples (11.00–13.95 s). The apple and stem are one collision body,
so the backend queries identify finger-pad force on that combined body rather
than a separately measured stem-only force.

## Full 14-second pad-force record

Each cell is **index pad / thumb pad** mean force magnitude in N over that
one-second interval. Every interval contains 20 samples at 0.05 s spacing;
the mean includes zero-force samples when contact is absent.

| Time (s) | SuperDex (N) | Isaac (N) | MuJoCo (N) |
| --- | ---: | ---: | ---: |
| 0–1 | 0 / 0 | 0 / 0 | 0 / 0 |
| 1–2 | 0 / 0 | 0 / 0 | 0 / 0 |
| 2–3 | 0 / 0 | 0 / 0 | 0 / 0 |
| 3–4 | 0 / 0 | 0 / 0 | 0 / 0 |
| 4–5 | 0 / 0 | 0 / 0 | 0 / 0 |
| 5–6 | 3.314 / 3.298 | 1.246 / 1.082 | 0.742 / 0.979 |
| 6–7 | 9.845 / 9.849 | 5.707 / 5.498 | 0.926 / 1.049 |
| 7–8 | 7.938 / 7.862 | 4.773 / 4.519 | 0.625 / 0.709 |
| 8–9 | 7.117 / 7.116 | 4.090 / 3.997 | 0 / 0 |
| 9–10 | 7.087 / 7.088 | 4.082 / 3.991 | 0 / 0 |
| 10–11 | 7.055 / 7.059 | 4.082 / 3.992 | 0 / 0 |
| 11–12 | 7.020 / 7.028 | 4.083 / 3.993 | 0 / 0 |
| 12–13 | 6.987 / 6.999 | 4.083 / 3.993 | 0 / 0 |
| 13–14 | 6.955 / 6.970 | 4.082 / 3.992 | 0 / 0 |

At the 0.05 s sample resolution, SuperDex first records index/thumb pad
contact at 5.35/5.45 s and Isaac at 5.50/5.70 s. MuJoCo first records
index/thumb contact at 5.50/5.05 s and last records either pad at 7.35 s.

## Full 14-second force vectors

Each row is the mean force on the combined apple/stem body in **world XYZ, N**, over 20 samples in that one-second interval. Positive Z points upward. These are vector means, so their magnitudes can differ from the mean magnitudes above.

### SuperDex

| Time (s) | Index Fx | Index Fy | Index Fz | Thumb Fx | Thumb Fy | Thumb Fz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0–1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1–2 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2–3 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3–4 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4–5 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5–6 | -0.739 | -3.156 | +0.690 | +0.727 | +3.138 | -0.707 |
| 6–7 | -2.207 | -9.393 | +1.955 | +2.193 | +9.377 | -2.065 |
| 7–8 | -2.201 | -7.490 | +1.372 | +2.178 | +7.488 | +0.313 |
| 8–9 | -2.096 | -6.731 | +0.978 | +2.096 | +6.728 | +0.985 |
| 9–10 | -2.089 | -6.703 | +0.971 | +2.089 | +6.703 | +0.980 |
| 10–11 | -2.081 | -6.671 | +0.967 | +2.081 | +6.671 | +0.994 |
| 11–12 | -2.074 | -6.639 | +0.953 | +2.074 | +6.639 | +1.008 |
| 12–13 | -2.066 | -6.608 | +0.940 | +2.066 | +6.608 | +1.021 |
| 13–14 | -2.056 | -6.579 | +0.928 | +2.057 | +6.579 | +1.034 |

### Isaac

| Time (s) | Index Fx | Index Fy | Index Fz | Thumb Fx | Thumb Fy | Thumb Fz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0–1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1–2 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2–3 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3–4 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4–5 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5–6 | -0.260 | -1.167 | +0.334 | +0.200 | +1.026 | -0.279 |
| 6–7 | -1.338 | -5.296 | +1.656 | +1.005 | +5.203 | -1.465 |
| 7–8 | -0.111 | -4.438 | +1.622 | +0.060 | +4.414 | +0.115 |
| 8–9 | +0.302 | -3.906 | +1.174 | -0.302 | +3.906 | +0.788 |
| 9–10 | +0.305 | -3.901 | +1.164 | -0.304 | +3.900 | +0.788 |
| 10–11 | +0.310 | -3.900 | +1.163 | -0.310 | +3.900 | +0.798 |
| 11–12 | +0.310 | -3.900 | +1.166 | -0.310 | +3.901 | +0.796 |
| 12–13 | +0.310 | -3.900 | +1.167 | -0.310 | +3.901 | +0.796 |
| 13–14 | +0.310 | -3.900 | +1.164 | -0.310 | +3.899 | +0.798 |

### MuJoCo

| Time (s) | Index Fx | Index Fy | Index Fz | Thumb Fx | Thumb Fy | Thumb Fz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0–1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1–2 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2–3 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3–4 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4–5 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5–6 | -0.212 | -0.704 | +0.026 | +0.394 | +0.836 | -0.145 |
| 6–7 | -0.163 | -0.879 | -0.214 | +0.470 | +0.891 | -0.273 |
| 7–8 | -0.163 | -0.576 | +0.110 | +0.206 | +0.600 | +0.161 |
| 8–9 | 0 | 0 | 0 | 0 | 0 | 0 |
| 9–10 | 0 | 0 | 0 | 0 | 0 | 0 |
| 10–11 | 0 | 0 | 0 | 0 | 0 | 0 |
| 11–12 | 0 | 0 | 0 | 0 | 0 | 0 |
| 12–13 | 0 | 0 | 0 | 0 | 0 | 0 |
| 13–14 | 0 | 0 | 0 | 0 | 0 | 0 |

## Force direction in the requested lift and hold windows

| Window | Backend | Apple contact | Mean Fx | Mean Fy | Mean Fz | Mean \|F\| | Peak \|F\| |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 8–9 s lift | SuperDex | Index pad | -2.096 | -6.731 | 0.978 | 7.117 | 7.176 |
| 8–9 s lift | SuperDex | Thumb pad | 2.096 | 6.728 | 0.985 | 7.116 | 7.169 |
| 8–9 s lift | Isaac | Index pad | 0.302 | -3.906 | 1.174 | 4.090 | 4.103 |
| 8–9 s lift | Isaac | Thumb pad | -0.302 | 3.906 | 0.788 | 3.997 | 4.011 |
| 8–9 s lift | MuJoCo | Index pad | 0 | 0 | 0 | 0 | 0 |
| 8–9 s lift | MuJoCo | Thumb pad | 0 | 0 | 0 | 0 | 0 |
| 11–14 s hold | SuperDex | Index pad | -2.066 | -6.609 | 0.940 | 6.987 | 7.036 |
| 11–14 s hold | SuperDex | Thumb pad | 2.066 | 6.609 | 1.021 | 6.999 | 7.042 |
| 11–14 s hold | Isaac | Index pad | 0.310 | -3.900 | 1.166 | 4.083 | 4.090 |
| 11–14 s hold | Isaac | Thumb pad | -0.310 | 3.900 | 0.797 | 3.993 | 4.000 |
| 11–14 s hold | MuJoCo | Index pad | 0 | 0 | 0 | 0 | 0 |
| 11–14 s hold | MuJoCo | Thumb pad | 0 | 0 | 0 | 0 | 0 |

The two pads' mean vertical forces sum to **1.961 N in SuperDex** and **1.962 N
in Isaac** during the hold, close to the apple's 1.962 N weight. Both apples
remain lifted. MuJoCo's pads have lost contact by the lift window; the table
supplies a mean vertical force of **1.968 N during 8–9 s** and **1.974 N during
11–14 s**, and the apple stays near the table.

MuJoCo's last sampled index/thumb pad contact is at **7.35 s**. At 7.30 s, the
index pad force is `(-0.772, -1.848, +0.584) N` and the thumb pad force is
`(+0.567, +1.914, +1.184) N`. Their normal-force vertical components are
**-0.706 N** and **-0.416 N**; their friction vertical components are **+1.290 N**
and **+1.600 N**. Each pad's friction-force magnitude is approximately equal
to its normal-force magnitude, consistent with its effective sliding
coefficient of 1.0 being saturated at that instant. The net vertical force
from the two pads is 1.768 N, below the apple's weight. This explains why a
large upward friction component alone did not sustain this grasp; it does not
identify which contact geometry or solver setting caused the downward normal
directions.

SuperDex used its native FP64 `get_contact_force_from_actor_world` query at a
2 ms step and default 0.5 Coulomb coefficient. Isaac used its PhysX contact
matrix **plus** friction-anchor forces at a 2 ms step, with static and dynamic
coefficients of 2.0 on the apple and pads. MuJoCo used `mj_contactForce`,
rotated from its contact frame and signed for force on the apple, at a 1 ms
step and default sliding coefficient of 1.0. The MuJoCo probe stepped the
same compiled task model and PD target schedule directly with `mj_step` to
read native contact forces; it did not read the UniSim adapter's worker data.
The engines also use different contact geometry and solver formulations, so
these forces are observations from their respective runs, not calibrated
material measurements. Sampling every 0.05 s can miss shorter force spikes.
None of these 14 s runs validates the later release and regrasp.

Local raw captures (ignored by Git):
`results/apple-contact-force/superdex/metrics.json`,
`results/apple-contact-force/isaac.json`, and
`results/apple-contact-force/mujoco.json`.

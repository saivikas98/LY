import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
 
# --- Web App Configuration ---
st.set_page_config(page_title="Lyophilization Simulator", layout="wide")
st.title("Lyophilization Batch Heterogeneity Simulator")
st.markdown("Simulates the primary to secondary drying transition, comparing Edge vs. Center vials.")
 
# --- UI Sidebar Controls ---
st.sidebar.header("Process Parameters")
T_shelf_C = st.sidebar.slider("Shelf Temp (°C)", -40.0, 20.0, -5.0, step=0.5)
P_mTorr = st.sidebar.slider("Chamber Pressure (mTorr)", 10.0, 300.0, 100.0, step=5.0)
 
st.sidebar.header("Formulation & Equipment")
Kv_edge_val = st.sidebar.slider("Edge Kv (1e-4 cal/s cm² K)", 3.5, 10.0, 5.5, step=0.1)
Ar_val = st.sidebar.slider("Cake Specific Resistance (Ar)", 1.0, 15.0, 5.0, step=0.5)
 
# --- Physical Constants ---
A_v, A_p = 17.3, 15.2
L_fill, rho_ice, epsilon = 3.3, 0.92, 0.95
dH_sub, k_ice = 676, 0.0055
m_total = A_p * rho_ice * epsilon * L_fill
 
def vp_ice(T_K):
    return np.exp(24.01849 - 6144.96 / T_K)
 
# Cache the heavy math solver so the web app stays snappy if inputs haven't changed
@st.cache_data
def simulate_batch_extremes(T_shelf_C, P_mTorr, Kv_edge_val, Ar_val):
    T_shelf = T_shelf_C + 273.15
    P_chamber = P_mTorr / 1000.0
    Kv_arr = np.array([Kv_edge_val * 1e-4, 3.5e-4])
    R0, Br = 0.5, 0.2
 
    def ode_system(t, y):
        m_sub, Tb = y[0:2], y[2:4]
        dm_dt, dTb_dt = np.zeros(2), np.zeros(2)
        L_d = m_sub / (A_p * rho_ice * epsilon)
        L_ice = L_fill - L_d
        
        for i in range(2):
            if L_ice[i] > 0:
                def energy_balance(Ti):
                    R_th = (1.0 / Kv_arr[i]) + (L_ice[i] / k_ice)
                    qin = A_v * (T_shelf - Ti) / R_th
                    Rp = R0 + (Ar_val * L_d[i]) / (1.0 + Br * L_d[i])
                    qout = A_p * (vp_ice(Ti) - P_chamber) / Rp * dH_sub
                    return qin - qout
                
                try:
                    Ti = root_scalar(energy_balance, bracket=[200, T_shelf]).root
                    R_th = (1.0 / Kv_arr[i]) + (L_ice[i] / k_ice)
                    qin = A_v * (T_shelf - Ti) / R_th
                    Rp = R0 + (Ar_val * L_d[i]) / (1.0 + Br * L_d[i])
                    
                    dm_dt[i] = A_p * (vp_ice(Ti) - P_chamber) / Rp
                    Tb_target = T_shelf - qin / (A_v * Kv_arr[i])
                    dTb_dt[i] = (Tb_target - Tb[i]) / 10.0
                except ValueError:
                    dm_dt[i], dTb_dt[i] = 0, (T_shelf - Tb[i]) / 10.0
            else:
                dm_dt[i], dTb_dt[i] = 0, (T_shelf - Tb[i]) / 500.0
        return np.concatenate((dm_dt, dTb_dt))
 
    y0 = [0.0, 0.0, T_shelf - 20, T_shelf - 20]
    sol = solve_ivp(ode_system, (0, 72 * 3600), y0, method='BDF', max_step=1800)
    
    ice_mass = np.maximum(m_total - sol.y[0:2], 0)
    return sol.t / 3600, ice_mass, sol.y[2:4] - 273.15
 
# --- Run Simulation & Render ---
with st.spinner("Integrating PDEs..."):
    t_hours, ice_mass, Tb_C = simulate_batch_extremes(T_shelf_C, P_mTorr, Kv_edge_val, Ar_val)
 
col1, col2 = st.columns(2)
 
with col1:
    st.subheader("Product Temperature Profiles")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(t_hours, Tb_C[0], 'r-', label='Edge Vial ($T_b$)')
    ax1.plot(t_hours, Tb_C[1], 'b-', label='Center Vial ($T_b$)')
    ax1.axhline(T_shelf_C, color='k', linestyle='--', alpha=0.5, label='Shelf Setpoint')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_xlabel('Time (Hours)')
    ax1.legend()
    ax1.grid(True, linestyle=':')
    st.pyplot(fig1)
 
with col2:
    st.subheader("Ice Mass Sublimation")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(t_hours, ice_mass[0], 'r-', label='Edge Ice Mass')
    ax2.plot(t_hours, ice_mass[1], 'b-', label='Center Ice Mass')
    ax2.set_ylabel('Remaining Ice (g)')
    ax2.set_xlabel('Time (Hours)')
    ax2.legend()
    ax2.grid(True, linestyle=':')
    st.pyplot(fig2)

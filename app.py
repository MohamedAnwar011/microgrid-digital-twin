import streamlit as st
import pandapower as pp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# =====================================================================
# 1. CONFIGURATION & MODEL LOADING
# =====================================================================
st.set_page_config(page_title="Digital Twin Microgrid", layout="wide")

WINDOW_SIZE = 15
FAULT_NAMES = ['Normal', 'Grid Sag', 'Feeder Overload', 'Gen Outage', 'Load Spike']
FEATURE_COLS = (
    ['total_load_mw', 'solar_gen_mw', 'diesel_gen_mw',
     'battery_power_mw', 'battery_soc', 'ext_grid_power_mw']
    + [f'voltage_bus{i}_pu' for i in range(1, 10)]
)

@st.cache_resource
def load_digital_twin_brain():
    model_path = os.path.join('outputs', 'microgrid_best_model.pkl')
    scaler_path = os.path.join('outputs', 'microgrid_scaler.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("Model files not found! Make sure 'microgrid_best_model.pkl' and 'microgrid_scaler.pkl' are in the 'outputs' folder.")
        st.stop()
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

ai_model, ai_scaler = load_digital_twin_brain()

# =====================================================================
# 2. INTERACTIVE PHYSICAL SIMULATION
# =====================================================================
@st.cache_data(show_spinner=False)
def run_interactive_simulation(sag_depth, overload_mult, spike_mult, starting_soc):
    """Runs a 1-day pandapower simulation based on user UI parameters."""
    rng = np.random.default_rng(42)
    net = pp.create_empty_network(f_hz=50)
    b = [pp.create_bus(net, vn_kv=0.4, name=f"Bus {i+1}") for i in range(9)]
    pp.create_ext_grid(net, bus=b[0], vm_pu=1.0, name="Grid Connection")

    line_data = [(0,3,0.1),(3,4,0.1),(4,5,0.1),(2,5,0.1),(5,6,0.1),(6,7,0.1),(1,7,0.1),(7,8,0.1),(8,3,0.1)]
    for src, dst, length in line_data:
        pp.create_line(net, from_bus=b[src], to_bus=b[dst], length_km=length, std_type="NAYY 4x50 SE")

    pp.create_load(net, bus=b[4], p_mw=0.04, q_mvar=0.01)
    pp.create_load(net, bus=b[6], p_mw=0.03, q_mvar=0.01)
    pp.create_load(net, bus=b[8], p_mw=0.03, q_mvar=0.01)
    pp.create_sgen(net, bus=b[7], p_mw=0.10, q_mvar=0.0)
    pp.create_sgen(net, bus=b[1], p_mw=0.0,  q_mvar=0.0)
    pp.create_storage(net, bus=b[3], p_mw=0.0, max_e_mwh=0.10, soc_percent=starting_soc)

    total_minutes = 1440
    time_index = pd.date_range("2026-04-01 00:00", periods=total_minutes, freq="1min")

    m = np.arange(1440)
    base_solar = np.zeros(1440)
    daylight = (m >= 360) & (m <= 1080)
    base_solar[daylight] = np.sin(np.pi * (m[daylight] - 360) / 720)
    base_res = 0.3 + 0.5 * np.exp(-0.5 * ((m - 480) / 60 )**2) + 0.7 * np.exp(-0.5 * ((m - 1200) / 90 )**2)
    base_com = 0.2 + 0.8 * np.exp(-0.5 * ((m - 780) / 240)**2)

    battery_soc = float(starting_soc)
    battery_cap, battery_max, diesel_ramp, current_diesel_kw = 100.0, 50.0, 10.0, 0.0

    # Inject one of each fault type across the day
    FAULT_SCHEDULE = [(200, 10, 1), (500, 10, 2), (800, 10, 3), (1100, 10, 4)]
    training_data = []

    for t in range(total_minutes):
        minute_of_day = t

        current_fault_code = 0
        for centre, half_dur, code in FAULT_SCHEDULE:
            if centre - half_dur <= minute_of_day < centre + half_dur:
                current_fault_code = code
                break

        grid_v = sag_depth if current_fault_code == 1 else 1.0
        net.ext_grid.at[0, 'vm_pu'] = float(np.clip(grid_v + rng.normal(0, 0.005), 0.5, 1.05))

        is_overload = (current_fault_code == 2)
        is_gen_outage = (current_fault_code == 3)
        is_spike = (current_fault_code == 4)

        noise = rng.normal(1.0, 0.03)
        com_mult = overload_mult if is_overload else 1.0
        res_mult = overload_mult if is_overload else spike_mult if is_spike else 1.0

        p_com = 0.04 * base_com[minute_of_day] * noise * com_mult
        p_res1 = 0.03 * base_res[minute_of_day] * noise * res_mult
        p_res2 = 0.03 * base_res[minute_of_day] * noise * res_mult

        net.load.at[0, 'p_mw'] = p_com
        net.load.at[1, 'p_mw'] = p_res1
        net.load.at[2, 'p_mw'] = p_res2

        net.sgen.at[0, 'p_mw'] = 0.10 * base_solar[minute_of_day]

        if is_gen_outage:
            current_diesel_kw = 0.0
        else:
            delta = 30.0 - current_diesel_kw
            current_diesel_kw += float(np.clip(delta, -diesel_ramp, diesel_ramp))
        net.sgen.at[1, 'p_mw'] = current_diesel_kw / 1000.0

        # EMS
        net.storage.at[0, 'p_mw'] = 0.0
        try:
            pp.runpp(net, verbose=False)
            grid_power_kw = net.res_ext_grid.at[0, 'p_mw'] * 1000.0
        except pp.powerflow.LoadflowNotConverged:
            grid_power_kw = 0.0

        if grid_power_kw < -1.0 and battery_soc < 100:
            charge_kw = min(abs(grid_power_kw), battery_max)
            net.storage.at[0, 'p_mw'] = -charge_kw / 1000.0
            battery_soc += (charge_kw * 0.95 / 60.0) / battery_cap * 100.0
        elif grid_power_kw > 1.0 and battery_soc > 10:
            discharge_kw = min(grid_power_kw, battery_max)
            net.storage.at[0, 'p_mw'] = discharge_kw / 1000.0
            battery_soc -= (discharge_kw / 0.95 / 60.0) / battery_cap * 100.0
        else:
            net.storage.at[0, 'p_mw'] = 0.0

        battery_soc = float(np.clip(battery_soc, 0.0, 100.0))
        net.storage.at[0, 'soc_percent'] = battery_soc

        converged = True
        try:
            pp.runpp(net, verbose=False)
        except pp.powerflow.LoadflowNotConverged:
            converged = False

        row = {
            'timestamp': time_index[t],
            'total_load_mw': net.load['p_mw'].sum(),
            'solar_gen_mw': net.sgen.at[0, 'p_mw'],
            'diesel_gen_mw': current_diesel_kw / 1000.0,
            'battery_power_mw': net.storage.at[0, 'p_mw'],
            'battery_soc': battery_soc,
            'fault_code': current_fault_code,
            'ext_grid_power_mw': net.res_ext_grid.at[0, 'p_mw'] if converged else np.nan,
        }
        for i in range(9):
            row[f'voltage_bus{i+1}_pu'] = net.res_bus.at[i, 'vm_pu'] if converged else np.nan
        training_data.append(row)

    return pd.DataFrame(training_data)

# =====================================================================
# 3. AI INFERENCE (ROLLING WINDOW)
# =====================================================================
def run_ai_inference(df, scaler, model):
    """Processes physical data through the rolling window buffer for prediction."""
    df_filled = df.ffill()
    scaled_data = scaler.transform(df_filled[FEATURE_COLS])
    
    predictions = np.zeros(len(df))
    # We skip the first 15 minutes because the AI needs a full window to start
    for i in range(len(scaled_data) - WINDOW_SIZE):
        window = scaled_data[i : i + WINDOW_SIZE]
        flat_window = window.reshape(1, -1)
        pred = model.predict(flat_window)
        predictions[i + WINDOW_SIZE] = pred[0]
        
    return predictions

# =====================================================================
# 4. STREAMLIT UI LAYOUT
# =====================================================================
st.title("⚡ Digital Twin Command Center: Fault Detection")
st.markdown("Use the sidebar to change the physical parameters of the microgrid. See if the pre-trained AI can successfully detect the faults under your custom stress-test conditions.")

with st.sidebar:
    st.header("1. Define System Physics")
    ui_soc = st.slider("Starting Battery SOC (%)", 0, 100, 50)
    ui_sag = st.slider("Grid Sag Depth (p.u.)", 0.50, 0.95, 0.80, help="How far does the grid voltage drop during a sag event?")
    
    st.header("2. Define Fault Thresholds")
    ui_overload = st.slider("Feeder Overload Multiplier", 1.5, 4.0, 2.5, help="How much does total load spike during an overload?")
    ui_spike = st.slider("Residential Spike Multiplier", 2.0, 6.0, 4.0)

    run_btn = st.button("RUN SCADA SIMULATION & AI INFERENCE", use_container_width=True, type="primary")

if run_btn:
    with st.spinner("Simulating physics and running AI inference..."):
        # 1. Generate new physical data
        live_df = run_interactive_simulation(ui_sag, ui_overload, ui_spike, ui_soc)
        
        # 2. Run the AI models
        live_df['ai_prediction'] = run_ai_inference(live_df, ai_scaler, ai_model)

    # 3. Plot Results
    st.success("Simulation Complete. AI Inference Applied.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Physical SCADA Feed (Voltages)")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(live_df['timestamp'], live_df['voltage_bus4_pu'], label="Bus 4 Voltage", color="#e74c3c")
        ax.plot(live_df['timestamp'], live_df['voltage_bus8_pu'], label="Bus 8 Voltage", color="#2c3e50")
        ax.axhline(0.95, color='red', linestyle=':', alpha=0.5)
        ax.set_ylabel("Voltage (p.u.)")
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("Physical SCADA Feed (Power)")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(live_df['timestamp'], live_df['total_load_mw']*1000, label="Total Load (kW)", color="#d0021b")
        ax2.plot(live_df['timestamp'], live_df['ext_grid_power_mw']*1000, label="Grid Import (kW)", color="#4a90d9")
        ax2.set_ylabel("Power (kW)")
        ax2.legend()
        st.pyplot(fig2)

    st.subheader("AI Fault Detection Performance")
    fig3, ax3 = plt.subplots(figsize=(12, 3))
    ax3.plot(live_df['timestamp'], live_df['fault_code'], label="Actual Physical Fault", color="black", linewidth=4, alpha=0.3)
    ax3.plot(live_df['timestamp'], live_df['ai_prediction'], label="AI Model Prediction", color="red", linestyle="--", linewidth=2)
    ax3.set_yticks(range(5))
    ax3.set_yticklabels(FAULT_NAMES)
    ax3.set_ylabel("System Status")
    ax3.legend(loc="upper left")
    st.pyplot(fig3)
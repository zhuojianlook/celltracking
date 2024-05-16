import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import numpy as np
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors

st.set_page_config(layout="wide")

# Helper functions used in the main function
if 'graph' not in st.session_state:
    st.session_state['graph'] = nx.DiGraph()
if 'depth_counters' not in st.session_state:
    st.session_state['depth_counters'] = defaultdict(lambda: defaultdict(int))

def authenticate_gspread(json_key):
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(json_key, scope)
    gc = gspread.authorize(credentials)
    return gc

def calculate_pd(cells_start, cells_end):
    if cells_start > 0 and cells_end > 0:
        return np.log10(cells_end / cells_start) / np.log10(2)
    else:
        return np.nan

def calculate_cpd(pd_values):
    return np.cumsum(pd_values)

def extract_cell_lines(nodes):
    cell_lines = set()
    for node in nodes:
        parts = node.split('P')
        if parts:
            cell_lines.add(parts[0])
    return list(cell_lines)

def load_data(sheet_url, gc):
    worksheet = gc.open_by_url(sheet_url).sheet1
    expected_headers = ['Node', 'Parent', 'Date', 'Vessel Type', 'Cells Start', 'Cells End', 'Notes', 'Media Change Interval', 'Media Change', 'Color', 'Media Change Count']
    
    data = worksheet.get_all_records(expected_headers=expected_headers)
    df = pd.DataFrame(data)
    
    for header in expected_headers:
        if header not in df.columns:
            df[header] = None  # Default to None for missing columns to handle in downstream processing

    # Convert and fill missing values appropriately
    df['Cells Start'] = pd.to_numeric(df['Cells Start'], errors='coerce')
    df['Cells End'] = pd.to_numeric(df['Cells End'], errors='coerce')
    df['Media Change Interval'] = pd.to_numeric(df['Media Change Interval'], errors='coerce').fillna(84)
    df['Media Change'] = pd.to_datetime(df['Media Change'], errors='coerce')
    df['Color'] = df['Color'].fillna('#d3d3d3')  # Default color if missing
    df['Media Change Count'] = pd.to_numeric(df['Media Change Count'], errors='coerce').fillna(0).astype(int)  # Default count to 0 if missing
    
    return df

def handle_cell_line_selection(unique_cell_lines):
    if 'selected_cell_lines' not in st.session_state or not st.session_state.selected_cell_lines:
        st.session_state.selected_cell_lines = st.multiselect('Select cell lines:', unique_cell_lines, key='cell_line_selector')
    else:
        st.session_state.selected_cell_lines = st.multiselect('Select cell lines:', unique_cell_lines, default=st.session_state.selected_cell_lines)

def save_data(df, sheet_url, gc):
    worksheet = gc.open_by_url(sheet_url).sheet1
    # Convert dates and handle missing data
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
    df['Media Change'] = pd.to_datetime(df['Media Change'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
    df.fillna('', inplace=True)
    df.replace([np.inf, -np.inf], '', inplace=True)
    
    # Ensure all expected columns exist and initialize them if not
    expected_columns = ['Media Change', 'Color', 'Media Change Count', 'Media Change Interval']
    for column in expected_columns:
        if column not in df.columns:
            if column == 'Media Change Count':
                df[column] = 0  # Initialize count with 0 for all rows
            elif column == 'Media Change Interval':
                df[column] = 84  # Set default media change interval
            else:
                df[column] = ''  # Initialize other string columns with empty strings

    # Update Google Sheet with new data
    try:
        current_data = worksheet.get_all_values()
        if not current_data or current_data == [[]]:  # If the sheet is empty, include headers
            headers = df.columns.values.tolist()
            worksheet.update([headers] + df.values.tolist(), value_input_option='USER_ENTERED')
        else:
            range_to_clear = f"A2:{chr(65 + len(df.columns) - 1)}{len(current_data)+1}"
            worksheet.batch_clear([range_to_clear])
            worksheet.append_rows(df.values.tolist(), value_input_option='USER_ENTERED')
    except Exception as e:
        print(f"Failed to save data: {e}")

def add_nodes(parent_node, num_children, creation_datetime, vessel_types, num_cells_start, num_cells_end_parent, notes, media_change_interval):
    if isinstance(creation_datetime, str):
        creation_datetime = datetime.strptime(creation_datetime, "%Y-%m-%d %H:%M:%S")

    # Check if the parent node already exists in the graph
    if parent_node not in st.session_state['graph']:
        # Initialize new parent node with a depth of -1 if it does not exist
        st.session_state['graph'].add_node(parent_node, date=creation_datetime, depth=-1, num_cells_end=num_cells_end_parent)
    else:
        # Update the cell end count for the existing parent node
        st.session_state['graph'].nodes[parent_node]['num_cells_end'] = num_cells_end_parent

    base_name = parent_node.split('P')[0] if 'P' in parent_node else parent_node
    current_depth = st.session_state['graph'].nodes[parent_node]['depth']

    next_depth = current_depth + 1

    if base_name not in st.session_state['depth_counters']:
        st.session_state['depth_counters'][base_name] = defaultdict(int)

    max_index = max([int(node.split('.')[1]) for node, data in st.session_state['graph'].nodes(data=True) if data['depth'] == next_depth and node.startswith(base_name)], default=-1) + 1

    child_index = max_index - 1

    for i in range(num_children):
        child_index = max_index + i
        child_node = f"{base_name}P{next_depth}.{child_index}"
        st.session_state['graph'].add_node(
            child_node,
            date=creation_datetime,
            depth=next_depth,
            vessel_type=vessel_types[i],
            num_cells_start=num_cells_start[i],
            notes=notes[i],
            media_change_interval=media_change_interval,
            Color=calculate_node_color(None, media_change_interval)  # Set initial color
        )
        st.session_state['graph'].add_edge(parent_node, child_node)

    st.session_state['depth_counters'][base_name][next_depth] = child_index + 1

def draw_graph():
    # Define node shapes by vessel type
    vessel_shapes = {
        'T75': 'o',  # Circle
        'T25': 's',  # Square
        'T125': 'p',  # Pentagon
        '12 well plate': 'h',  # Hexagon
        '6 well plate': '^',  # Triangle up
        'Cryovial': 'v',  # Triangle down
        'Unknown': 'd'  # Diamond
    }

    # Use multipartite layout as the base
    pos = nx.multipartite_layout(st.session_state['graph'], subset_key="depth")

    # Sort nodes in each subset to minimize edge crossings
    subsets = {}
    for node, data in st.session_state['graph'].nodes(data=True):
        depth = data['depth']
        if depth not in subsets:
            subsets[depth] = []
        subsets[depth].append(node)

    # Calculate new positions by sorting nodes within each depth
    new_pos = {}
    num_layers = len(subsets)
    max_width = max(len(subset) for subset in subsets.values()) if subsets else 0
    vertical_spacing_factor = 2

    for depth, nodes in subsets.items():
        sorted_nodes = sorted(nodes)
        if depth > min(subsets.keys()):
            prev_layer = subsets[depth - 1]
            sorted_nodes = sorted(sorted_nodes, key=lambda x: -sum(1 for pred in st.session_state['graph'].predecessors(x) if pred in prev_layer))

        for index, node in enumerate(sorted_nodes):
            new_pos[node] = (depth, -index * max_width * vertical_spacing_factor / len(sorted_nodes))

    # Determine the figure size dynamically
    max_nodes_in_layer = max(len(layer) for layer in subsets.values()) if subsets else 0
    fig_width = max(20, num_layers * 3)
    fig_height = max(10, max_nodes_in_layer * vertical_spacing_factor)

    plt.figure(figsize=(fig_width, fig_height), dpi=300)
    for node, data in st.session_state['graph'].nodes(data=True):
        node_color = data.get('Color', '#d3d3d3')  # Default to grey if Color is not found

        nx.draw_networkx_nodes(st.session_state['graph'], new_pos, nodelist=[node],
                               node_size=7000,
                               node_color=node_color,
                               node_shape=vessel_shapes.get(data.get('vessel_type', 'Unknown'), 'o'))
    nx.draw_networkx_edges(st.session_state['graph'], new_pos, arrowstyle='-|>', arrowsize=10)
    labels = {
        node: f"{node}\nDate: {data['date']}\nVessel: {data.get('vessel_type', 'Unknown')}\nCells start: {data.get('num_cells_start', 'N/A')}\nCells end: {data.get('num_cells_end', 'N/A')}\nNotes: {data.get('notes', '')}"
        for node, data in st.session_state['graph'].nodes(data=True)
    }
    nx.draw_networkx_labels(st.session_state['graph'], new_pos, labels=labels, font_size=9, font_color="black")

    plt.savefig("graph.png", dpi=300)
    plt.close()
    st.image("graph.png", use_column_width=True)

def reconstruct_graph(df):
    st.session_state['graph'].clear()  # Clear the existing graph
    st.session_state['depth_counters'].clear()

    for _, row in df.iterrows():
        node = row['Node']
        depth = -1 if 'P' not in node else int(node.split('P')[1].split('.')[0])
        
        st.session_state['graph'].add_node(
            node,
            date=row.get('Date', 'N/A'),
            vessel_type=row.get('Vessel Type', 'Unknown'),
            num_cells_start=row.get('Cells Start', 'N/A'),
            num_cells_end=row.get('Cells End', 'N/A'),
            notes=row.get('Notes', ''),
            depth=depth,
            media_change_interval=row.get('Media Change Interval', 84),
            Media_Change=row.get('Media Change', None),  # Ensure Media Change is captured
            Color=row.get('Color', '#d3d3d3')  # Add the Color attribute
        )

    for _, row in df.iterrows():
        child_node = row['Node']
        parent_node = row['Parent']
        if pd.notnull(parent_node) and parent_node in st.session_state['graph']:
            st.session_state['graph'].add_edge(parent_node, child_node)
    
    for node, data in st.session_state['graph'].nodes(data=True):
        last_change_datetime = pd.to_datetime(data.get('Media_Change'), errors='coerce')
        media_change_interval = data.get('media_change_interval', 84)
        color = calculate_node_color(last_change_datetime, media_change_interval)
        st.session_state['graph'].nodes[node]['Color'] = color

def plot_graphs(cpd_data, dt_data):
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    for cell_line, values in cpd_data.items():
        valid_values = [value for value in values if isinstance(value, (int, float)) and not np.isnan(value)]
        if len(valid_values) > 0:
            x = np.arange(len(valid_values))
            if len(valid_values) > 3:
                x_new = np.linspace(x.min(), x.max(), 300)
                spl = make_interp_spline(x, valid_values, k=3)
                y_smooth = spl(x_new)
                plt.plot(x_new, y_smooth, label=f'{cell_line}')
            else:
                plt.plot(x, valid_values, label=f'{cell_line}', linestyle='--')
            plt.scatter(x, valid_values, marker='o')
    ax1 = plt.gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title('Cumulative Population Doublings (cPD) vs. Passage Number')
    plt.xlabel('Passage Number')
    plt.ylabel('cPD')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    for cell_line, values in dt_data.items():
        valid_values = [value for value in values if isinstance(value, (int, float)) and not np.isnan(value)]
        if len(valid_values) > 0:
            x = np.arange(len(valid_values))
            if len(valid_values) > 3:
                try:
                    x_new = np.linspace(x.min(), x.max(), 300)
                    spl = make_interp_spline(x, valid_values, k=3)
                    y_smooth = spl(x_new)
                    plt.plot(x_new, y_smooth, label=f'{cell_line}')
                except Exception as e:
                    print(f"Interpolation error for {cell_line}: {e}")
                    plt.plot(x, valid_values, label=f'{cell_line}', linestyle='--')
            else:
                plt.plot(x, valid_values, label=f'{cell_line}', linestyle='--')
            plt.scatter(x, valid_values, marker='o')
    ax2 = plt.gca()
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title('Doubling Time (DT) vs. Passage Number')
    plt.xlabel('Passage Number')
    plt.ylabel('DT (hours/PD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt.gcf())

def update_media_change(node, media_change_datetime, sheet_url, gc):
    df = load_data(sheet_url, gc)  # Load existing data
    mask = df['Node'] == node

    if not mask.any():
        st.error("Selected node not found in the data.")
        return None
    
    if mask.sum() > 1:
        st.error(f"More than one entry found for node {node}. Please check the data for duplicates.")
        return None

    if 'Media Change Count' not in df.columns:
        df['Media Change Count'] = 0  # Initialize if column does not exist

    # Safe to use .item() as mask is guaranteed to have exactly one True value
    current_count = df.loc[mask, 'Media Change Count'].fillna(0).astype(int)
    new_count = current_count + 1

    df.loc[mask, 'Media Change'] = media_change_datetime.strftime('%Y-%m-%d %H:%M:%S')
    df.loc[mask, 'Color'] = '#FF007F'  # Set color to red-pink
    df.loc[mask, 'Media Change Count'] = new_count  # Update the count

    save_data(df, sheet_url, gc)

    # Update node attributes in the graph as well
    if node in st.session_state['graph']:
        st.session_state['graph'].nodes[node]['Media Change'] = media_change_datetime.strftime('%Y-%m-%d %H:%M:%S')
        st.session_state['graph'].nodes[node]['Color'] = '#FF007F'
        st.session_state['graph'].nodes[node]['Media Change Count'] = new_count.iloc[0]  # Convert to scalar

    st.success(f"Media change for {node} updated successfully. Color set to red-pink, and media change count is now {new_count.iloc[0]}.")
    return df
def save_data_to_sheet(sheet_url, gc):
    worksheet = gc.open_by_url(sheet_url).sheet1
    headers = ['Node', 'Parent', 'Date', 'Vessel Type', 'Cells Start', 'Cells End', 'Notes', 'Media Change Interval', 'Media Change', 'Color', 'Media Change Count']
    
    # Create DataFrame from graph data
    nodes_data = [
        {
            'Node': node,
            'Parent': list(st.session_state['graph'].predecessors(node))[0] if list(st.session_state['graph'].predecessors(node)) else None,
            'Date': data.get('date', ''),
            'Vessel Type': data.get('vessel_type', 'Unknown'),
            'Cells Start': data.get('num_cells_start', 'N/A'),
            'Cells End': data.get('num_cells_end', 'N/A'),
            'Notes': data.get('notes', ''),
            'Media Change Interval': data.get('media_change_interval', 84),
            'Media Change': data.get('Media Change', ''),
            'Color': data.get('Color', ''),
            'Media Change Count': data.get('Media Change Count', 0),
        } for node, data in st.session_state['graph'].nodes(data=True)
    ]

    new_data_df = pd.DataFrame(nodes_data)

    # Get existing data from the sheet
    existing_data = worksheet.get_all_records()
    if existing_data:
        existing_df = pd.DataFrame(existing_data)
    else:
        existing_df = pd.DataFrame(columns=headers)

    # Ensure all columns exist in the existing data
    for column in headers:
        if column not in existing_df.columns:
            existing_df[column] = None

    # Merge existing data with new data, updating existing entries and adding new ones
    existing_df.set_index('Node', inplace=True)
    new_data_df.set_index('Node', inplace=True)

    # Update the 'Cells End' values for parent nodes
    for node, data in st.session_state['graph'].nodes(data=True):
        if node in existing_df.index:
            existing_df.at[node, 'Cells End'] = data.get('num_cells_end', 'N/A')

    for column in ['Media Change', 'Color', 'Media Change Count']:
        new_data_df[column] = new_data_df[column].combine_first(existing_df[column])
    updated_df = existing_df.combine_first(new_data_df).reset_index()

    # Convert dates and handle missing data
    updated_df['Date'] = pd.to_datetime(updated_df['Date'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
    updated_df['Media Change'] = pd.to_datetime(updated_df['Media Change'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
    updated_df.fillna('', inplace=True)
    updated_df.replace([np.inf, -np.inf], '', inplace=True)

    # Ensure all expected columns exist and initialize them if not
    for column in headers:
        if column not in updated_df.columns:
            if column == 'Media Change Count':
                updated_df[column] = 0  # Initialize count with 0 for all rows
            elif column == 'Media Change Interval':
                updated_df[column] = 84  # Set default media change interval
            else:
                updated_df[column] = ''  # Initialize other string columns with empty strings

    # Convert the DataFrame to list of lists for batch update
    updated_data = [headers] + updated_df.values.tolist()

    # Batch update Google Sheet with new data
    try:
        # Clear the worksheet before updating
        worksheet.clear()
        # Update the worksheet with all rows in a single request
        worksheet.update(updated_data, value_input_option='USER_ENTERED')
    except Exception as e:
        st.error(f"Failed to save data: {e}")



def calculate_node_color(last_change_datetime, media_change_interval):
    if last_change_datetime is None or pd.isnull(last_change_datetime):
        return '#d3d3d3'  # Default to grey if no media change data
    now = datetime.now()
    time_diff = (now - last_change_datetime).total_seconds() / 3600  # Time difference in hours

    if time_diff >= media_change_interval:
        return '#FFFF00'  # Yellow color after full interval has passed

    fraction = time_diff / media_change_interval
    red_pink = mcolors.to_rgba('#FF007F')
    yellow = mcolors.to_rgba('#FFFF00')

    new_color = [red_pink[i] * (1 - fraction) + yellow[i] * fraction for i in range(4)]
    return mcolors.to_hex(new_color)

def main():
    st.title('Cell Culture Tracking Program')
    if 'should_draw_graph' not in st.session_state:
        st.session_state['should_draw_graph'] = False
    default_base_node = ''
    if 'reset' in st.session_state and st.session_state.reset:
        default_base_node = ''
    else:
        default_base_node = st.session_state.get('new_base_node', '')

    st.sidebar.title("Instructions")
    st.sidebar.markdown(r"""                 
        ### How to Use This Application
        To effectively use this application, follow these steps:
        1. Upload your Google service account JSON key.
        2. Enter the URL of your Google Sheet that is shared with the service account.
        3. Use the controls to add culture steps and define passage parameters.
        4. Click 'Add to Graph' to create and visualize the graph.
        5. Use 'Save Data to Sheet' to save your graph data to Google Sheets.
        6. Use 'Load Data from Sheet' to load and reconstruct the graph from Google Sheets.
        7. Press 'Refresh' if you need to refresh the session state, e.g., after adding a cell line.

        ### Best Practices for Cell Culture Calculations and Methodologies
        #### Understanding Population Doubling (PD) and Cumulative Population Doubling (cPD)
        - **Population Doubling (PD)** is used as a better measure of cellular age in culture versus passage number.
        - **Cumulative Population Doubling (cPD)** represents the total number of times the cell population has doubled since the onset of the culture. Cells from different cell lines with the same cumulative population doubling can be considered to be the same biological age.
        - **Parameters to keep constant:**
            *** First Use Case *** In order for the calculations to work well: Type and size of flasks used should be identical across cell lines, flasks should be seeded at a fixed cell density and should be passaged after a fixed amount of time for accurate generation of cumulative population doubling data. e.g. Every 7days. Volume of media used e.g. 0.1mL/cm2 should be kept constant. Number of hours between media changes should be kept constant. 
            In this scenario, a slower growing cell line will have a lower cPD than a faster growing cell line. It is also possible to identify the fastest or slowest growing passage stage within each cell line by looking at the gradient between the previous passage and the current passage number in the generated graph. 
            It is expected that as the passage number increases, the gradient becomes more horizontal as the effect of senecence comes into play, the shape of the graph will therefore be sigmoidal in shape. For experimental purposes, if the cPD of cell lines are similar, they can be understood to be of the same biological age (not taking into account the selection stress that passage creates and therefore a possible limitation of comparing different cell lines of same cPD but different passage level. Note that this culture strategy may be more relavent for homogenous cell cultures such as where the effect of passage has a known minimal effect on the heterogenity of the cellular population, however a determination of growth potential between different cell lines or donors is required.
            *** Second Use Case *** Alernatively, for primary cultures of a heterogenous tissue source, instead of a fixed amount of time in passage, it is also possible to fix an endpoint for each passage e.g. 70% confluency. In this scenario, the cPD vs Passage Number graph gradient would be best maintained as constant slope for all cell lines, with deviations attributing to operator error in determining cell density before passage. It is also best practice to match the passage with an integral multiple the population doubling (easiest case 2X). E.g. Passage when population has doubled, tripled or quadrupled.  Therefore experience with correlating confluency via light microscopy and expected number of cells is required from the operator. 
            With this strategy, multiple cell lines from heterogenous sources are more comparable as the selective pressure of passage is kept constant. The primary limitation is that cell lines with differing senescence are likely to reach the same passage point at differing times. 
            With this strategy, passage level and cPD are equivalent measures of cellular biological age. Therefore, a separate graph for growth rate (doubling time vs passage number) is also automatically generated. Passage levels with faster expansion rates would have a lower doubling time. For primary cultures, it is expected that as the passage levels increase the doubling time likewise increases or follows a normal distribution with left skew e.g. peaking at P2. For experimental purposes, because the PDs are fixed per passage, differing population doubling times at a chosen passage level can be used to infer proliferative rates that can be correlated to characteristics in source material such as donor age. 
        - **Caveats:** This program is not of sufficent granularity to measure generation time, which is the interval from one point in the cell division cycle to the same point in the cycle on division later. This should not be confused with the population doubling time, which is derived from the total cell count of a population and therefore averages different generation times and is influenced by nongrowing and dying cells. 
        - **Media Change:** Due to the challenges of keeping track of media changes, this program allows the user to set a minimum number of hours required for media changes for each passage. The color of the flask will then change depending on the number of hours to the next media change (Pink to Yellow) for a visual representation. Media change may be necessary due to stability of specific components within the media, depletion and should be kept constant. Ultimately this entire process should be machine automated but engineering costs and complexity are a major factor in most laboratories.
        #### Calculating Population Doubling (PD)
        To calculate the population doubling during a specific passage:
        1. **Formula**:
        $$
        PD = \frac{\log\left(\frac{\text{Number of cells at harvest}}{\text{Number of cells seeded}}\right)}{\log(2)}
        $$

        - Convert the ratio of harvested cells to seeded cells into a logarithmic scale using log base 10, and then divide by log(2) to convert it into base 2.
        - This calculation gives the number of times the cell population has doubled during the passage.

        #### Calculating Doubling Time (DT)
        Doubling time reflects how quickly cells double in number during a culture period.
        1. **Formula**:
        $$
        DT = \frac{\text{Time in culture}}{PD}
        $$

        - Divide the total culture duration by the population doubling to find how often the cells doubled during that time.

        #### Tracking Cumulative Population Doubling (cPD)
        - To obtain the cumulative population doubling, add the PD of each passage sequentially.
        - Example progression:
        - Passage 1 (P1): PD = 3.5
        - Passage 2 (P2): PD = 3.0 (Cumulative PD = 6.5)
        - Passage 3 (P3): PD = 2.5 (Cumulative PD = 9.0)
        - This series provides a running total of cell doublings, useful for assessing cell line vitality and comparability across different lines.

        #### Plotting Growth Curves
        - Plot cPD against the passage number to visualize the growth or expansion curve of the cell line.
        - This curve helps compare the growth rates and senescence stages across different cell lines.

        #### Practical Tips for Cell Culture
        - Always maintain the recommended minimum seeding density specific to each cell line.
        - While not strictly required for calculations, standardize seeding density for experiments to ensure reproducibility
        - Harvest cells based on fixed time points or confluence levels (e.g., 80-90% confluence).
        - Document all relevant parameters:
        - Days in culture.
        - Seeding and harvest cell count numbers (for PD and DT calculations).
        - Maintain a consistent type/number of flasks and media volumes used (important for scaling and reproducibility). e.g. 0.1 ml of media per cm²              

        ### Setting Up Prerequisites
        To set up prerequisites for using this application, follow these steps:
        Approximate video guide: (https://www.youtube.com/watch?v=fxGeppjO0Mg)
        1. **Create a Google service account JSON Key**:
            - **Step 1**: Create a [Google Cloud Project](https://console.cloud.google.com/welcome?project=cell-culture-tracking).
            - Go to the Google Cloud Console.
            - Sign in with your Google account if you haven't already done so.
            - Click on the project dropdown at the top of the page and then click on "New Project".
            - Enter a project name and select a billing account if necessary.
            - Click on "Create".
            - **Step 2**: Enable Google Sheets API.
            - Make sure the new project is selected.
            - Navigate to "APIs & Services > Dashboard".
            - Click on "+ ENABLE APIS AND SERVICES".
            - Search for "Google Sheets API" and click on it.
            - Click on "Enable".
            - **Step 3**: Create a Service Account.
            - Go to "IAM & Admin > Service accounts".
            - Click on "Create Service Account".
            - Enter a name and description for the service account.
            - Click on "Create and Continue".
            - You can skip granting this service account access to the project and click "Continue".
            - Click "Done" to finish creating the service account.
            - **Step 4**: Create the JSON Key.
            - Find the service account you just created in the list and click on it.
            - Go to the "Keys" tab.
            - Click on "Add Key" and choose "JSON" from the dropdown menu.
            - Your browser will download a JSON file containing the private key. Keep it safe.
        2. Create a Google sheet with your normal Google account and share the Google sheet with the service account email with 'write' privileges. The service account email can be found in the JSON file and should look like 'example@example.iam.gserviceaccount.com'.
        3. Note that if you want to retroactively modify the date in google sheets, you have to follow the same format, otherwise saving to sheet will overwrite a blank value. Note that starting from a completely blank google sheet is the intended start use case. Please save your data redundantly
        """)

    # Connection and sheet handling
    uploaded_file = st.file_uploader("Upload Google service account JSON key", type="json")
    sheet_url = st.text_input("Enter the URL of your Google Sheet")
    gc = None
    if uploaded_file and sheet_url:
        json_key = json.load(uploaded_file)
        gc = authenticate_gspread(json_key)
        if gc:
            st.success("Connected to Google Sheets successfully!")

    if gc and sheet_url:
        if st.button('Load Data from Sheet', key='load_data_from_sheet'):
            df = load_data(sheet_url, gc)
            reconstruct_graph(df)
            st.success("Data loaded and graph reconstructed successfully from Google Sheets.")
            st.session_state['should_draw_graph'] = True  # Set flag to draw graph

        df = load_data(sheet_url, gc)
        node_options = df['Node'].dropna().unique().tolist()
        selected_node = st.selectbox('Select a Node to Change Media:', [''] + node_options)
        if selected_node:
            media_change_date = st.date_input("Select Date for Media Change:", value=datetime.now())
            media_change_time = st.time_input("Select Time for Media Change:", value=datetime.now().time())
            media_change_datetime = datetime.combine(media_change_date, media_change_time)
            if st.button('Change Media', key='change_media'):
                updated_df = update_media_change(selected_node, media_change_datetime, sheet_url, gc)
                if updated_df is not None:
                    st.session_state['latest_data'] = updated_df
                    st.session_state['should_draw_graph'] = True  # Set flag to draw graph

        if st.button('Save Data to Sheet', key='save_data_to_sheet'):
            save_data_to_sheet(sheet_url, gc)
            st.session_state['should_draw_graph'] = True  # Set flag to draw graph

    num_children = st.number_input('Enter Number of Child Vessels', min_value=0, step=1, format="%d")
    vessel_options = ['T75', 'T25', 'T125', '12 well plate', '6 well plate', 'Cryovial']
    vessel_selections = [st.selectbox(f'Vessel type for child {i+1}:', vessel_options, key=f'vessel_{i}') for i in range(num_children)]
    num_cells_start = [st.number_input(f'Total start cell number seeded for child {i+1}:', min_value=0, format="%d", key=f'cells_start_{i}') for i in range(num_children)]
    notes = [st.text_input(f'Notes for child {i+1}:', key=f'notes_{i}') for i in range(num_children)]
    num_cells_end_parent = st.number_input('Total end cell number for Parent vessel', min_value=0, format="%d", key='cells_end_parent')
    creation_date = st.date_input('Select Date of Passage', value=datetime.today())
    # Time selection with session state management
    if 'time_of_passage' not in st.session_state:
        st.session_state['time_of_passage'] = datetime.now().time()
    creation_time = st.time_input('Select Time of Passage', value=st.session_state['time_of_passage'])
    st.session_state['time_of_passage'] = creation_time
    full_creation_datetime = datetime.combine(creation_date, creation_time)

    existing_nodes = [node for node in st.session_state['graph']]
    base_node_selection = st.selectbox('Select an existing vessel', [""] + existing_nodes, index=0, key="base_node_selection")
    new_base_node = st.text_input('Or enter a new lineage name', value=default_base_node, key="new_base_node")
    media_change_interval = st.number_input('Set Media Change Interval (hours)', min_value=0, value=84, step=1, format="%d", key='media_change_interval')

    if st.button('Add Entry', key='add_entry'):
        base_node = new_base_node if new_base_node else base_node_selection
        if base_node:
            add_nodes(base_node, num_children, full_creation_datetime, vessel_selections, num_cells_start, num_cells_end_parent, notes, media_change_interval) 
            st.session_state['should_draw_graph'] = True

    if st.button('Refresh', key='refresh'):
        st.session_state.reset = True
        st.rerun()

    if 'reset' in st.session_state and st.session_state.reset:
        st.session_state.reset = False

    if gc and sheet_url:
        df = load_data(sheet_url, gc)
        unique_cell_lines = extract_cell_lines(df['Node'])
        selected_cell_lines = st.multiselect('Select cell lines:', unique_cell_lines, key='cell_line_selection')

        if 'selected_flasks' not in st.session_state:
            st.session_state['selected_flasks'] = {}

        for cell_line in selected_cell_lines:
            cell_line_data = df[df['Node'].str.contains(f'^{cell_line}P')]
            passage_dict = defaultdict(list)
            for node in cell_line_data['Node']:
                parts = node.split('P')
                if len(parts) > 1 and '.' in parts[1]:
                    passage = int(parts[1].split('.')[0])
                    passage_dict[passage].append(node)

            sorted_passages = sorted(passage_dict.keys())
            if cell_line not in st.session_state['selected_flasks']:
                st.session_state['selected_flasks'][cell_line] = {}

            for passage in sorted_passages:
                flask_options = passage_dict[passage]
                selected_flask_key = f"{cell_line}_passage_{passage}"
                if selected_flask_key not in st.session_state['selected_flasks'][cell_line]:
                    st.session_state['selected_flasks'][cell_line][selected_flask_key] = flask_options[0] if flask_options else None
                current_selection = st.session_state['selected_flasks'][cell_line][selected_flask_key]
                st.session_state['selected_flasks'][cell_line][selected_flask_key] = st.selectbox(
                    f'Select flask for {cell_line} passage {passage}:',
                    flask_options,
                    key=selected_flask_key,
                    index=flask_options.index(current_selection) if current_selection in flask_options else 0
                )

        if st.button('Calculate PD, DT, and Generate Graphs', key='calculate_graphs_button'):
            pd_data = {}
            cpd_data = {}
            time_in_culture_data = {}
            dt_data = {}
            max_passage = 0

            for cell_line in selected_cell_lines:
                pd_values = []
                time_in_culture_values = []
                dt_values = []
                
                for _, flask in st.session_state['selected_flasks'][cell_line].items():
                    flask_data = df[df['Node'] == flask].iloc[0]
                    cells_start = flask_data['Cells Start']
                    cells_end = flask_data['Cells End']
                    pd_value = calculate_pd(cells_start, cells_end)
                    pd_values.append(pd_value)
                    
                    if flask in st.session_state['graph']:
                        child_nodes = list(st.session_state['graph'].successors(flask))
                        if child_nodes:
                            child_node = child_nodes[0]
                            parent_datetime = pd.to_datetime(st.session_state['graph'].nodes[flask]['date'])
                            child_datetime = pd.to_datetime(st.session_state['graph'].nodes[child_node]['date'])
                            time_diff = child_datetime - parent_datetime
                            time_in_culture_hours = time_diff.total_seconds() / 3600
                            time_in_culture_values.append(time_in_culture_hours)
                            if pd_value != 0:
                                dt_values.append(time_in_culture_hours / pd_value)
                            else:
                                dt_values.append(None)
                        else:
                            time_in_culture_values.append(None)
                            dt_values.append(None)
                
                pd_data[cell_line] = pd_values
                cpd_data[cell_line] = np.cumsum(pd_values)
                time_in_culture_data[cell_line] = time_in_culture_values
                dt_data[cell_line] = dt_values
                max_passage = max(max_passage, len(pd_values))

            max_passages = [len(pd_data[cl]) for cl in selected_cell_lines]
            all_pd = [pd for cl in selected_cell_lines for pd in pd_data[cl]]
            all_times = [time for cl in selected_cell_lines for time in time_in_culture_data[cl]]
            all_dts = [dt for cl in selected_cell_lines for dt in dt_data[cl]]
            all_passages = [list(range(passages)) for passages in max_passages]
            flat_passages = [item for sublist in all_passages for item in sublist]
            all_cell_lines = [[cl] * len(pd_data[cl]) for cl in selected_cell_lines]
            flat_cell_lines = [item for sublist in all_cell_lines for item in sublist]

            df_calculated = pd.DataFrame({
                'Cell Line': flat_cell_lines,
                'Passage Number': flat_passages,
                'Population Doublings': all_pd,
                'Time in Culture (hrs)': all_times,
                'Doubling Time (hrs/PD)': all_dts
            })
            st.dataframe(df_calculated)  # Display the DataFrame in the Streamlit interface
            plot_graphs(cpd_data, dt_data)

    if st.session_state['graph'].nodes() and st.session_state['should_draw_graph']:
        draw_graph()
        st.session_state['should_draw_graph'] = False  # Reset the flag to prevent unnecessary redraws

if __name__ == "__main__":
    main()
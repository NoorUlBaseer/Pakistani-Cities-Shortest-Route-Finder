import csv
import pandas as pd
import math
import heapq
import streamlit as st
import time

def data_cleanup(csv_file):  # Function to clean up data and extract Pakistani cities
    df = pd.read_csv(csv_file)

    pak_cities = df[df['country'] == 'Pakistan'].copy()  # Filter for Pakistani cities
    pak_cities = pak_cities[['city', 'lat', 'lng']]  # Select relevant columns

    pak_cities.to_csv('pak_cities.csv', index=False)


def calculate_distance_km(lat1, lon1, lat2, lon2):  # Haversine formula to calculate distance between two lat/lon points
    # Step 1: Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Step 2: Compute differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Step 3: Apply Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Step 4: Earth’s radius in kilometers
    R = 6371  

    # Step 5: Total distance
    distance = R * c
    return distance


def build_graph(file):  # Function to build graph from cleaned data
    cities = []  # List to hold city data

    with open(file, newline='') as f:  # Read cleaned CSV file
        reader = csv.DictReader(f)
        
        for row in reader:
            cities.append({
                "name": row["city"],
                "lat": float(row["lat"]),
                "lon": float(row["lng"])
            })

    # Build edges based on threshold
    threshold_km = 200
    edges = []  # List to hold edges
    n = len(cities)

    for i in range(n):  # Iterate through each pair of cities
        for j in range(i+1, n):
            d = calculate_distance_km(cities[i]["lat"], cities[i]["lon"],
                                      cities[j]["lat"], cities[j]["lon"])

            if d <= threshold_km:  # If distance is within threshold, create an edge
                edges.append((cities[i]["name"], cities[j]["name"], round(d,1)))

    return cities, edges


def create_adjacency_list(cities, edges):  # Function to create adjacency list from edges
    adj = { city["name"]: [] for city in cities }  # Initialize adjacency list

    for u, v, dist in edges:  # Populate adjacency list with edges
        adj[u].append((v, dist))
        adj[v].append((u, dist))

    return adj


def dijkstra(graph, start_city, end_city):  # Dijkstra's algorithm to find shortest path
    queue = [(0, start_city, [start_city])]  # Priority queue initialized with start city
    visited = set()  # Set to track visited nodes

    while queue:  # While there are nodes to process
        (current_dist, current_node, path) = heapq.heappop(queue)  # Get node with smallest distance

        if current_node in visited:  # If already visited, skip
            continue
        else:  # If not visited
            visited.add(current_node)  # Mark as visited
        
        if current_node == end_city:  # If destination reached
            return path, current_dist
        
        if current_node in graph:  # Explore neighbors
            for neighbor, weight in graph[current_node]:
                if neighbor not in visited:  # If neighbor not visited
                    heapq.heappush(queue, (current_dist + weight, neighbor, path + [neighbor]))  # Add to queue
    
    return None, float('inf')  # If no path found


def main():
    @st.cache_data  # Cache the graph loading function
    def load_graph():
        # Step 1: Clean up data
        print("Cleaning up data...")
        data_cleanup('worldcities.csv')
        print("Data cleaned and saved to pak_cities.csv")

        print()

        # Step 2: Build graph
        print("Building graph...")
        cities, edges = build_graph('pak_cities.csv')
        print(f"Graph built with {len(cities)} cities and {len(edges)} edges.")

        print()

        # Step 3: Create adjacency list
        print("Creating adjacency list...")
        adj_list = create_adjacency_list(cities, edges)
        print("Adjacency list created.")

        return adj_list

    # Streamlit UI
    st.title("Pakistani Cities Shortest Route Finder")
    st.markdown("This application finds the shortest route between two cities in Pakistan using **Dijkstra's algorithm**.")
    st.markdown("---")
    
    graph = load_graph()  # Load the graph
    cities = sorted(list(graph.keys()))  # List of cities for selection

    col1, col2 = st.columns(2)  # Two dropdowns for city selection

    # Add placeholder option to the cities list
    placeholder = "Select a City"
    cities.insert(0, placeholder)

    with col1:
        start_city = st.selectbox("Select the starting city", cities)  # Dropdown for starting city

    with col2:
        end_city = st.selectbox("Select the destination city", cities)  # Dropdown for destination city

    if st.button("Calculate"):  # Button to trigger calculation
        if start_city == end_city:  # Check if cities are the same
            st.warning("Please select two different cities.")
        elif start_city == placeholder or end_city == placeholder:  # Check if cities are selected
            st.warning("Please select both starting and destination cities.")
        else:  # Calculate shortest path
            with st.spinner("Calculating shortest path..."):  # Show spinner while calculating
                path, distance = dijkstra(graph, start_city, end_city)  # Run Dijkstra's algorithm
                time.sleep(2)  # Simulate processing time
            
            if path:  # If a path is found
                st.success("Route Found!")
                
                st.metric(label="Total Distance (km)", value=f"{distance} km")  # Display total distance

                # Display the route
                st.write("### Route:")
                st.info(" → ".join(path))
            else:  # If no path is found
                st.error("No path found between the specified cities.")

    # Footer
    st.markdown("---")
    st.caption("Developed by Noor Ul Baseer")


if __name__ == "__main__":
    main()

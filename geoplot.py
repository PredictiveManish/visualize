"""
geoplot.py
----------

This visualization renders a 3-D plot of the data given the state
trajectory of a simulation, and the path of the property to render.

It generates an HTML file that contains code to render the plot
using Cesium Ion, and the GeoJSON file of data provided to the plot.

An example of its usage is as follows:

```py
from agent_torch.visualize import GeoPlot

# create a simulation
# ...

# create a visualizer
engine = GeoPlot(config, {
  cesium_token: "...",
  step_time: 3600,
  coordinates = "agents/consumers/coordinates",
  feature = "agents/consumers/money_spent",
})

# visualize in the runner-loop
for i in range(0, num_episodes):
  runner.step(num_steps_per_episode)
  engine.render(runner.state_trajectory)
```
"""

import re
import json

import pandas as pd
import numpy as np

from string import Template
from agent_torch.core.helpers import get_by_path

geoplot_template = """
<!doctype html>
<html lang="en">
	<head>
		<meta charset="UTF-8" />
		<meta
			name="viewport"
			content="width=device-width, initial-scale=1.0"
		/>
		<title>Cesium Time-Series Heatmap Visualization</title>
		<script src="https://cesium.com/downloads/cesiumjs/releases/1.95/Build/Cesium/Cesium.js"></script>
		<link
			href="https://cesium.com/downloads/cesiumjs/releases/1.95/Build/Cesium/Widgets/widgets.css"
			rel="stylesheet"
		/>
		<style>
			#cesiumContainer {
				width: 100%;
				height: 100%;
			}
		</style>
	</head>
	<body>
		<div id="cesiumContainer"></div>
		<script>
			// Your Cesium ion access token here
			Cesium.Ion.defaultAccessToken = '$accessToken'

			// Create the viewer
			const viewer = new Cesium.Viewer('cesiumContainer')

			function interpolateColor(color1, color2, factor) {
				const result = new Cesium.Color()
				result.red = color1.red + factor * (color2.red - color1.red)
				result.green =
					color1.green + factor * (color2.green - color1.green)
				result.blue = color1.blue + factor * (color2.blue - color1.blue)
				result.alpha = '$visualType' == 'size' ? 0.2 :
					color1.alpha + factor * (color2.alpha - color1.alpha)
				return result
			}

			function getColor(value, min, max) {
				const factor = (value - min) / (max - min)
				return interpolateColor(
					Cesium.Color.BLUE,
					Cesium.Color.RED,
					factor
				)
			}

			function getPixelSize(value, min, max) {
				const factor = (value - min) / (max - min)
				return 100 * (1 + factor)
			}

			function processTimeSeriesData(geoJsonData) {
				const timeSeriesMap = new Map()
				let minValue = Infinity
				let maxValue = -Infinity

				geoJsonData.features.forEach((feature) => {
					const id = feature.properties.id
					const time = Cesium.JulianDate.fromIso8601(
						feature.properties.time
					)
					const value = feature.properties.value
					const coordinates = feature.geometry.coordinates

					if (!timeSeriesMap.has(id)) {
						timeSeriesMap.set(id, [])
					}
					timeSeriesMap.get(id).push({ time, value, coordinates })

					minValue = Math.min(minValue, value)
					maxValue = Math.max(maxValue, value)
				})

				return { timeSeriesMap, minValue, maxValue }
			}

			function createTimeSeriesEntities(
				timeSeriesData,
				startTime,
				stopTime
			) {
				const dataSource = new Cesium.CustomDataSource(
					'AgentTorch Simulation'
				)

				for (const [id, timeSeries] of timeSeriesData.timeSeriesMap) {
					const entity = new Cesium.Entity({
						id: id,
						availability: new Cesium.TimeIntervalCollection([
							new Cesium.TimeInterval({
								start: startTime,
								stop: stopTime,
							}),
						]),
						position: new Cesium.SampledPositionProperty(),
						point: {
							pixelSize: '$visualType' == 'size' ? new Cesium.SampledProperty(Number) : 10,
							color: new Cesium.SampledProperty(Cesium.Color),
						},
						properties: {
							value: new Cesium.SampledProperty(Number),
						},
					})

					timeSeries.forEach(({ time, value, coordinates }) => {
						const position = Cesium.Cartesian3.fromDegrees(
							coordinates[0],
							coordinates[1]
						)
						entity.position.addSample(time, position)
						entity.properties.value.addSample(time, value)
						entity.point.color.addSample(
							time,
							getColor(
								value,
								timeSeriesData.minValue,
								timeSeriesData.maxValue
							)
						)

						if ('$visualType' == 'size') {
						  entity.point.pixelSize.addSample(
  							time,
  							getPixelSize(
  								value,
  								timeSeriesData.minValue,
  								timeSeriesData.maxValue
  							)
  						)
						}
					})

					dataSource.entities.add(entity)
				}

				return dataSource
			}

			// Example time-series GeoJSON data
			const geoJsons = $data

			const start = Cesium.JulianDate.fromIso8601('$startTime')
			const stop = Cesium.JulianDate.fromIso8601('$stopTime')

			viewer.clock.startTime = start.clone()
			viewer.clock.stopTime = stop.clone()
			viewer.clock.currentTime = start.clone()
			viewer.clock.clockRange = Cesium.ClockRange.LOOP_STOP
			viewer.clock.multiplier = 3600 // 1 hour per second

			viewer.timeline.zoomTo(start, stop)

			for (const geoJsonData of geoJsons) {
				const timeSeriesData = processTimeSeriesData(geoJsonData)
				const dataSource = createTimeSeriesEntities(
					timeSeriesData,
					start,
					stop
				)
				viewer.dataSources.add(dataSource)
				viewer.zoomTo(dataSource)
			}
		</script>
	</body>
</html>
"""

# Helper function to retrieve the value of a variable from a nested dictionary using a path string
# For example, "agents/consumers/coordinates" would access state["agents"]["consumers"]["coordinates"]
def read_var(state, var):
    return get_by_path(state, re.split("/", var))


class GeoPlot:
    def __init__(self, config, options):
        """
        Initialize the GeoPlot visualization class.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary containing simulation metadata (name, num_episodes, num_steps_per_episode).
        
        options : dict
            A dictionary containing:
            - cesium_token: API token for Cesium Ion (required for map visualization)
            - step_time: Time interval in seconds between simulation steps
            - coordinates: Path to the coordinates data in the state object (e.g., "agents/consumers/coordinates")
            - feature: Path to the feature/property data in the state object (e.g., "agents/consumers/money_spent")
            - visualization_type: Type of visualization ('color' for heatmap or 'size' for bubble map)
        """
        self.config = config
        (
            self.cesium_token,     # Cesium API token for visualization
            self.step_time,        # Time step between states in seconds
            self.entity_position,  # Path to the position data in the state
            self.entity_property,  # Path to the property data in the state
            self.visualization_type, # Type of visualization (color or size)
        ) = (
            options["cesium_token"],
            options["step_time"],
            options["coordinates"],
            options["feature"],
            options.get("visualization_type", "color"),  # Default to color if not specified
        )

    def render(self, state_trajectory):
        """
        Process the simulation data and generate GeoJSON and HTML files for visualization.
        
        Parameters:
        -----------
        state_trajectory : list
            A list of simulation states over time. The format is:
            [
                [  # Episode 0
                    {  # Step 0 final state
                        "agents": {
                            "consumers": {
                                "coordinates": [[lat1, lon1], [lat2, lon2], ...],  # List of coordinates
                                "money_spent": [100, 200, ...]  # Corresponding values for each coordinate
                            }
                        }
                    }
                ],
                [  # Episode 1
                    {  # Step 0 final state
                        "agents": {
                            "consumers": {
                                "coordinates": [[lat1, lon1], [lat2, lon2], ...],
                                "money_spent": [150, 250, ...]
                            }
                        }
                    }
                ],
                ...
            ]
            
            Example mock data structure:
            ```
            # Mock state trajectory data (points and polygons)
            coords = [
                [[-71.03215355, 42.37951895], [-71.13215355, 42.42951895], [-71.03215355, 42.42951895], [-71.03215355, 42.37951895]],
                [[-71.23215355, 42.36951895], [-71.03215355, 42.42051895], [-71.03215355, 42.46851895], [-71.23215355, 42.22341895]],
            ]

            # Each row is a time interval, and each column is the value for the i-th polygon in the coords array
            values = [
                [100, 40],
                [200, 980],
                [800, 240],
                [50, 680],
            ]

            # Simulate the state trajectory
            state_trajectory = [
                [{"mock_coordinates_path": coords, "mock_feature_path": values[i]}]
                for i in range(len(values))
            ]
            ```
        """
        
        coords, values = [], []  # Initialize lists for coordinates and values
        name = self.config["simulation_metadata"]["name"]  # Get simulation name from config
        geodata_path, geoplot_path = f"{name}.geojson", f"{name}.html"  # Define output file paths
        
        # Extract coordinates and property values from the simulation states
        # We're iterating through episodes and getting the final state of each episode
        for i in range(0, len(state_trajectory) - 1):
            final_state = state_trajectory[i][-1]  # Get the last state in the episode
            
            # Read position and property data from the final state using the provided paths
            # coords will be a list of [lat, lon] pairs for each entity
            coords = np.array(read_var(final_state, self.entity_position)).tolist()
            
            # values will be a list of lists, where each inner list contains property values for all entities at one time step
            values.append(
                np.array(read_var(final_state, self.entity_property)).flatten().tolist()
            )
            
        # Generate timestamps for the simulation based on step_time and metadata
        # This creates a sequence of timestamps starting from now and separated by step_time seconds
        start_time = pd.Timestamp.utcnow()
        timestamps = [
            start_time + pd.Timedelta(seconds=i * self.step_time)
            for i in range(
                self.config["simulation_metadata"]["num_episodes"]
                * self.config["simulation_metadata"]["num_steps_per_episode"]
            )
        ]
        
        # Create GeoJSON data for visualization
        # Each entity (e.g., consumer) will have its own GeoJSON FeatureCollection
        geojsons = []
        for i, coord in enumerate(coords):
            features = []  # List to store features for a single coordinate over time
            for time, value_list in zip(timestamps, values):
                features.append(
                    {
                        "type": "Feature",  # GeoJSON feature type
                        "geometry": {
                            "type": "Point",  # Point geometry with longitude and latitude (GeoJSON order)
                            "coordinates": [coord[1], coord[0]],  # Note: GeoJSON uses [lon, lat] order
                        },
                        "properties": {
                            "id": f"entity_{i}",  # Unique identifier for this entity
                            "value": value_list[i],  # Value associated with this entity at this time
                            "time": time.isoformat(),  # Timestamp in ISO format
                        },
                    }
                )
            geojsons.append({"type": "FeatureCollection", "features": features})  # Add features to GeoJSON
            
        # Write the GeoJSON data to a file
        with open(geodata_path, "w", encoding="utf-8") as f:
            json.dump(geojsons, f, ensure_ascii=False, indent=2)
        
        # Create an HTML visualization using the GeoJSON data and Cesium API
        # This substitutes variables in the HTML template with actual values
        tmpl = Template(geoplot_template)
        with open(geoplot_path, "w", encoding="utf-8") as f:
            f.write(
                tmpl.substitute(
                    {
                        "accessToken": self.cesium_token,  # Cesium API token
                        "startTime": timestamps[0].isoformat(),  # Simulation start time
                        "stopTime": timestamps[-1].isoformat(),  # Simulation end time
                        "data": json.dumps(geojsons),  # GeoJSON data as a string
                        "visualType": self.visualization_type,  # Visualization type (color or size)
                    }
                )
            )
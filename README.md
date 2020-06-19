# sumo-manhattan
Build up a manhattan model in SUMO

### Run SUMO from commandline window

1. To generate a network with dedicated number of lanes

```
netgenerate --grid --grid.number 5 --default.lanenumber 3  --output-file=net.net.xml
```

2. To generate traffic flows with OD

```
python /usr/share/sumo/tools/randomTrips.py -n net.net.xml -o flows.xml --begin 0 --end 1 --flows 100 --jtrrouter --trip-attributes 'departPos="random" departSpeed="max"'
```

3. To generate routes

```
jtrrouter -c manhattan.jtrrcfg
```

4. Run SUMO GUI

```
sumo-gui manhattan.sumocfg
```

### A simple DDoS detection/prevention system for SDN

Testbed is based on devstack and mininet. 
The jobs includes: 

1. Offline Training with scikit-learn
2. Get online sflow traffics with sflow-rt or pyshark
3. Prediction based on filtered features
4. Control ovs by calling REST API provided by OpenDayLight
5. Utils to get flow table, etc. 


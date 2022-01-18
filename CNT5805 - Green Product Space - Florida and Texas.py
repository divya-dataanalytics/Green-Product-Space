#!/usr/bin/env python
# coding: utf-8

# ## CNT5805 - Final Project Code
# ### Green Product Space - Florida and Texas
# 

# In[1]:


# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
pd.options.mode.chained_assignment = None 
from matplotlib.lines import Line2D
import scipy # required for kamada kawai layout


# ## Load data and clean up

# In[2]:


# read export data files
exports_US = pd.read_csv(r'D:\Sem3-Fall 2021\CNT5805-Network Science\Final Project\State Exports by HS Commodities-US.csv', skiprows = 4, usecols=range(3))
exports_2020 = pd.read_csv(r'D:\Sem3-Fall 2021\CNT5805-Network Science\Final Project\State Exports by HS Commodities-2020.csv', skiprows = 4, usecols=range(4))


exports_2019 = pd.read_csv(r'D:\Sem3-Fall 2021\CNT5805-Network Science\Final Project\State Exports by HS Commodities-2019.csv', skiprows = 4, usecols=range(4))
exports_2018 = pd.read_csv(r'D:\Sem3-Fall 2021\CNT5805-Network Science\Final Project\State Exports by HS Commodities-2018.csv', skiprows = 4, usecols=range(4))
exports_2017 = pd.read_csv(r'D:\Sem3-Fall 2021\CNT5805-Network Science\Final Project\State Exports by HS Commodities-2017.csv', skiprows = 4, usecols=range(4))
exports_2016 = pd.read_csv(r'D:\Sem3-Fall 2021\CNT5805-Network Science\Final Project\State Exports by HS Commodities-2016.csv', skiprows = 4, usecols=range(4))

# read green product list
green_prod = pd.read_csv(r'D:\Sem3-Fall 2021\CNT5805-Network Science\Final Project\GreenProductList.csv')

# read green product theme
gp_category = pd.read_csv(r'D:\Sem3-Fall 2021\CNT5805-Network Science\Final Project\Green Product Categories.csv')
gp_category = gp_category.rename(columns={"Code  ": "Medium", "Environmental theme or medium ": "Theme"})


# In[3]:


def data_cleanup(df):
    df['HSCode'] = df['Commodity'].str[:6].astype(str).astype(np.int64)
    df['Commodity'] = df['Commodity'].str[7:]
    df['Total Value ($US)'] = df['Total Value ($US)'].str.replace(',' , '').astype(np.int64)
    return df


# In[4]:


# clean up all export data files
file_list = [exports_US, exports_2020, exports_2019, exports_2018, exports_2017, exports_2016 ]
for file in file_list:
    file = data_cleanup(file)


# ## RCA (2016 - 2020)

# In[5]:


states = exports_2020['State'].unique()
years = exports_US['Time'].unique()
export_dfs = [exports_2016, exports_2017, exports_2018, exports_2019, exports_2020]


# In[6]:


# create a dictionary to store all years export files 
export_dict = {}
for i, year in enumerate(years):
    file_name = 'exports_'+ str(year)
    export_dict[year] = pd.DataFrame(export_dfs[i])


# In[7]:


# create a dictionary to store all years RCA    
RCA_df = {}

for year in years:
    df = export_dict[year]
    df['US total export'] = sum(exports_US[exports_US.Time == year]['Total Value ($US)'])
    df = df.rename(columns={"Total Value ($US)": "state export"})

    for state in states:
        df.loc[df.State == state, 'state total export'] = sum(df.loc[df.State == state, 'state export'])
        export_dict[year] = df
        
    gp_US = pd.merge(exports_US[exports_US.Time == year], green_prod, on = 'HSCode') 
    
    gp_states = pd.merge(df, green_prod, on = 'HSCode')
    gp_states = pd.merge(gp_states, gp_US, on = ['Time','HSCode', 'Medium','Commodity'])
    gp_states = gp_states.rename(columns={"Total Value ($US)": "US export"})
        
    gp_states['RCA'] = (gp_states['state export']/gp_states['state total export'])                        /(gp_states['US export']/gp_states['US total export'])
    
    RCA_df[year] = gp_states


# In[8]:


RCA_matrix = {}
for year in years:
    RCA_matrix[year] = RCA_df[year].set_index(['State', 'HSCode']).unstack('HSCode')['RCA']


# ## MCP matrix (2016 - 2020)

# In[9]:


MCP_matrix = {}
for year in years:
    df = RCA_matrix[year].fillna(0.0).applymap(lambda x: 1 if x >= 1.0 else 0.0)
    df = df.loc[:, (df != 0).any(axis=0)] 
    MCP_matrix[year] = df


# ## Proximity (2016 - 2020)

# In[10]:


def proximity(mcp_matrix):
    prod_list = list(mcp_matrix.columns) # sort list?

    # empty dataframe
    prox_matrix = pd.DataFrame(index=prod_list, columns=prod_list)
    prod_sum = mcp_matrix.sum(axis=0) # sum of countries that export a product competitively

    for i, prod_i in enumerate(prod_list):
        for j, prod_j in enumerate(prod_list):
            
            if i > j:  
                continue
            
            num = (mcp_matrix[prod_i] * mcp_matrix[prod_j]).sum() # ((RCA_TX_i * RCA_TX_j) + (RCA_FL_i * RCA_FL_j))
                                                                  #  => prob that both prod are exported competively  
            den = max(prod_sum[prod_i], prod_sum[prod_j])         # max(no. countries that exp prod i RCA>=1, 
                                                                  # no. countries tat exp prod j RCA>=1)  )
                                                                  # => prob that one prod is exported competively
            if den == 0:
                cond_prob = np.nan # or 0
            else:
                cond_prob = num / den
            
            prox_matrix.at[prod_i, prod_j] = cond_prob
        
    return prox_matrix


# In[11]:


proximity_dict = {}
for year in years:
    proximity_dict[year] = proximity(MCP_matrix[year])


# ## Green Product Space - Initial Networks

# In[12]:


def edges_list(prox_df, prox_value):
    edges = prox_df.unstack().dropna()
    edges = edges.reset_index()
    edges.columns = ["Prod1","Prod2","weight"]
    edges = edges[edges.Prod1 != edges.Prod2] 
    edges = edges[edges.weight != 0] 
    edges = edges[edges.weight >= prox_value] 
    return edges    


# In[13]:


def mst(edges, year):
    gps_network = nx.from_pandas_edgelist(edges, source="Prod1", target="Prod2", edge_attr=["weight"])
    mst = nx.maximum_spanning_tree(gps_network)
    #print("Number of nodes(mst):", mst.number_of_nodes())
    #print("Number of edges(mst):", mst.number_of_edges())
    
    fig = plt.figure(figsize=(50,20))
    ax = fig.gca()
    ax.text( 0.98, 0.9,
        "Nodes: " + str(mst.number_of_nodes()) + "\n"+\
        "Edges: " + str(mst.number_of_edges()),
        style='italic',
        verticalalignment = 'center', horizontalalignment = 'left',
        transform=ax.transAxes,
        #color='blue',
            fontsize=50)
    plt.title("GPS network using maximum spanning tree - "+ str(year), fontsize = 50)
    return nx.draw(mst, pos=nx.fruchterman_reingold_layout(mst),node_color = 'blue',)


# In[14]:


def initial_network(edges, year, prox_value, pos):
    gps_network = nx.from_pandas_edgelist(edges, source="Prod1", target="Prod2", edge_attr=["weight"])
   
    #print(gps_network.number_of_nodes())
    #print(gps_network.number_of_edges())
    
    fig = plt.figure(figsize=(70,50))
    ax = fig.gca()
    if prox_value == 0.0:
        plt.title("GPS network with all edges - "+ str(year), fontsize = 70)
    else:
        plt.title("GPS network with $ϕ$ >= " + str(prox_value) + " (" +str(year) + ")", fontsize = 70, y=1.0, pad=-70)
    if pos == 0:
        pos = nx.fruchterman_reingold_layout(gps_network, k = 0.5, seed = 143)
    else:
        pos == pos  
    
    ax.text( 0.98, 0.9,
        "Nodes: " + str(gps_network.number_of_nodes()) + "\n"+\
        "Edges: " + str(gps_network.number_of_edges()),
        style='italic',
        verticalalignment = 'center', horizontalalignment = 'left',
        transform=ax.transAxes,
        #color='blue',
        fontsize=70)
    
    nx.draw(gps_network, pos=pos, node_color = 'blue')
    return gps_network, pos


# In[15]:


years = [2016, 2017, 2018 , 2019, 2020]
print('\033[1m' , "Year", " Edges Count")

for year in years:
    edgeslist = edges_list(proximity_dict[year] , 0.0)
    print('\033[0m' , year, "   ", len(edgeslist))
    initial_network(edgeslist, year, 0.0, 0)
    mst(edgeslist, year)


# ## GPS for all years with proximity >= 0.25

# In[16]:


gps_network_graph = {}
gps_node_pos = {}
for year in years:
    edgeslist = edges_list(proximity_dict[year] , 0.25)
    gps_network_graph[year], gps_node_pos[year]  = initial_network(edgeslist, year, 0.25, 0)


# ## Different Layouts

# In[17]:


graph = gps_network_graph[2020]

fig = plt.figure(figsize = (50,50))
ax = fig.gca()
nx.draw_networkx(graph, pos= nx.spring_layout(graph, k = 0.5, seed = 143), 
                 node_size=200, width=1, node_color = 'blue',
                 with_labels = False)
ax.set_title("Green Product Space Network - Spring Layout", fontsize = 60, y=1.0, pad=-70)
plt.show()


# In[18]:


fig = plt.figure(figsize = (50,50))
ax = fig.gca()
nx.draw_networkx(graph, pos=nx.fruchterman_reingold_layout(graph, k = 0.5, seed = 143), #, iterations=100), 
                 node_size=200, width=1, node_color = 'blue',
                 with_labels = False)
ax.set_title("Green Product Space Network - Fruchterman Reingold Layout", fontsize = 60, y=1.0, pad=-70)
plt.show()


# In[19]:


fig = plt.figure(figsize = (50,50))
ax = fig.gca()
nx.draw_networkx(graph, pos=nx.kamada_kawai_layout(graph), 
                 node_size=200, width=1, node_color = 'blue',
                 with_labels = False)
ax.set_title("Green Product Space Network - Kamada Kawai Layout", fontsize = 60, y=1.0, pad=-70)
plt.show()


# ## Network at different proximity cutoff

# In[20]:


pos = gps_node_pos[2020]
edgeslist = edges_list(proximity_dict[2020] , 0.2)
a, b = initial_network(edgeslist, 2020, 0.2, pos)

edgeslist = edges_list(proximity_dict[2020] , 0.3)
a, b = initial_network(edgeslist, 2020, 0.3, pos)

edgeslist = edges_list(proximity_dict[2020] , 0.4)
a, b = initial_network(edgeslist, 2020, 0.4, pos)

edgeslist = edges_list(proximity_dict[2020] , 0.5)
a, b = initial_network(edgeslist, 2020, 0.5, pos)


# ## GPS - Texas and Florida

# In[21]:


def gps_state_year(state, year, graph, pos):
    # select the RCA df for a year and filter by state
    gp_state_year = RCA_df[year][RCA_df[year].State == state]
    
    # assign red color for all nodes
    gp_state_year['Node_color'] = "red"
    
    # assign green color for nodes with RCA >= 1
    gp_state_year.loc[(gp_state_year.RCA) >= 1, 'Node_color'] = 'green'
    
    # convert nodes list from graph to df
    nodes_list = pd.DataFrame(list(graph.nodes())).reset_index()
    nodes_list.columns = ["order", "HSCode"]
    
    # combine RCA df and nodes
    nodes_state_yr = gp_state_year.merge(nodes_list, how = "outer", on = "HSCode")
    nodes_state_yr = nodes_state_yr.sort_values(by = "order")
    
    nodes_state_yr = nodes_state_yr[nodes_state_yr['order'].notna()]

    # assign color for nodes not in list
    nodes_state_yr.loc[nodes_state_yr.Node_color.isnull(), 'Node_color'] = 'grey'
    
    count = len(gp_state_year.loc[(gp_state_year.RCA) >= 1])
    
    # plot
    fig = plt.figure(figsize = (50,50))
    ax = fig.gca()
    ax.text( 0.98, 0.9,
        "GP with RCA>=1 is " + str(count),
        style='italic',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=50)
        
    ax.legend(custom_dots, ['RCA<1', 'RCA>=1'],fontsize = 50)    
    plt.title("Product space network - " + state+ " " +str(year) , fontsize = 60, y=1.0, pad=-70)

    return nx.draw(graph, pos=pos,node_color = nodes_state_yr.Node_color.values)# node_size=100, width=1, with_labels = False,  )


# In[22]:


custom_dots =  [Line2D([0], [0], marker='o', color='w', label='RCA<1', markerfacecolor='r', markersize=25),
                Line2D([0], [0], marker='o', color='w', label='RCA>=1', markerfacecolor='g', markersize=25)]


# In[23]:


for year in years:
    for state in ['Texas', 'Florida']:
        gps_state_year(state, year, gps_network_graph[year], gps_node_pos[year])


# ## 1.	How different are the two states in green product space?

# In[24]:


for year in [2020]:
        RCA_1_TX = RCA_df[year][(RCA_df[year].State == 'Texas')  & (RCA_df[year].RCA >= 1)]
        RCA_1_TX = pd.merge(RCA_1_TX, gp_category, on = 'Medium')
        
        RCA_1_FL = RCA_df[year][(RCA_df[year].State == 'Florida')  & (RCA_df[year].RCA >= 1)]
        RCA_1_FL = pd.merge(RCA_1_FL, gp_category, on = 'Medium')
        
        TX_FL_1 = pd.merge((RCA_1_TX[['State', 'HSCode','Medium','Theme','RCA']]),                     (RCA_1_FL[['State', 'HSCode','Medium','Theme','RCA']]),                      how = "inner", on = 'HSCode') 
        TX_FL_1 = TX_FL_1.rename(columns={"Medium_y": "Medium", "Theme_y": "Theme"})

        print('\033[1m')
        print("Year:", year)
        print("GP produced competively by both FL and TX:",'\033[0m',len(TX_FL_1))
        TX_FL_df = pd.DataFrame({'FL&TX Count' : TX_FL_1.groupby(['Medium', 'Theme']).size()}).reset_index()
               
        TX_FL_2 = pd.merge((RCA_1_FL[['State', 'HSCode','Medium','Theme','RCA']]),                     (RCA_1_TX[['State', 'HSCode','Medium','Theme','RCA']]),                      how = "outer", on = 'HSCode')
        #TX_FL_2 = TX_FL_2[TX_FL_2.isna().any(axis=1)]

        #print(len(TX_FL_2))
        FL_df = TX_FL_2[TX_FL_2.State_x == 'Florida']
        print('\033[1m')
        print("GP produced competively by FL:",'\033[0m',  len(FL_df))
        FL_df = FL_df.rename(columns={"Medium_x": "Medium", "Theme_x": "Theme"})
        FL_df = pd.DataFrame({'FL Count' : FL_df.groupby(['Medium', 'Theme']).size()}).reset_index()

        TX_df = TX_FL_2[TX_FL_2.State_y == 'Texas']
        
        print('\033[1m')
        print("GP produced competively by TX:",'\033[0m',  len(TX_df))
        TX_df = TX_df.rename(columns={"Medium_y": "Medium", "Theme_y": "Theme"})
        TX_df = pd.DataFrame({'TX Count' : TX_df.groupby(['Medium', 'Theme']).size()}).reset_index()
        
        pd.DataFrame({'count' : TX_df.groupby(['Medium', 'Theme']).size()}).reset_index()      


# In[25]:


TX_FL_counts = pd.merge(pd.merge(TX_FL_df, TX_df, how = "outer"), FL_df, how = "outer")
TX_FL_counts[['FL&TX Count', 'TX Count']] = TX_FL_counts[['FL&TX Count', 'TX Count']].astype('Int32').fillna(0)
with pd.option_context('display.max_colwidth', None):
      display(TX_FL_counts)


# In[26]:


print(sum(TX_FL_counts['FL Count']))
print(sum(TX_FL_counts['TX Count']))
print(sum(TX_FL_counts['FL&TX Count']))


# In[27]:


RCA_1_TX[(RCA_1_TX.Medium == 'REP')]


# ## 2. Do the states show path dependency in green product evaluation?

# In[28]:


def relatedness(initial_year, final_year, state):
    initial_df = RCA_df[initial_year]
    final_df = RCA_df[final_year]
    
    # new green products
    new_gp_list = pd.merge(initial_df[(initial_df.State == state) & (initial_df.RCA < 0.5)],
                           final_df[(final_df.State == state) & (final_df.RCA >=1)], how ='inner', on = ["HSCode",'Commodity', 'State','Medium'])
    new_gp_list = new_gp_list[['State','HSCode','Commodity','Medium', 'Time_x','Time_y','RCA_x','RCA_y']]
    
    #new_gp_TX = new_gp_list[new_gp_list.State == 'Texas']
    RCA1 = MCP_matrix[initial_year][MCP_matrix[initial_year].index == state]
    RCA1 = RCA1.loc[:, (RCA1 != 0).any(axis=0)]
    
    relatedness = proximity_dict[initial_year]
    relatedness = relatedness.filter(items = new_gp_list.HSCode, axis = 0)
    relatedness = relatedness.filter(items = RCA1.columns)
    relatedness = relatedness.fillna(0)
    relatedness = relatedness.max(axis = 1)
    df = pd.DataFrame(relatedness, columns = ['Max proximity'])

    return df


# In[29]:


relatedness(2016, 2018, 'Florida')


# In[30]:


# verify
a = RCA_df[2016]
a[(a.State == 'Florida') & (a.HSCode == 851410)]


# In[31]:


# verify
a = RCA_df[2018]
a[(a.State == 'Florida') & (a.HSCode == 851410)]


# In[32]:


for year in [2016, 2017, 2018]:
    for state in ['Texas', 'Florida']:
        print("\nBase year:", year, "\nFinal year:", year+2, "\nState:", state)
        print(relatedness(year, year+2, state)) 


# In[33]:


for state in ['Texas']:
    resultTX = pd.Series(dtype='float')
    for year in [2016, 2017, 2018]:
        res = relatedness(year, year+2, state)
        res = res['Max proximity']#.reset_index(drop = 'True', in_place = 'True')
        resultTX = resultTX.append(res, ignore_index=True)
    print(resultTX)


# In[34]:


for state in ['Florida']:
    resultFL = pd.Series(dtype='float')
    for year in [2016, 2017, 2018]:
        res = relatedness(year, year+2, state)
        res = res['Max proximity'].reset_index(drop = 'True')
        resultFL = resultFL.append(res, ignore_index=True)
    print(resultFL)


# In[35]:


states = exports_2020['State'].unique()
result_final = pd.Series(dtype='float')
for state in states:
    
    result = pd.Series(dtype='float')
    for year in [2016, 2017, 2018]:
        res = relatedness(year, year+2, state)
        res = res['Max proximity']#.reset_index(drop = 'True', in_place = 'True')
        result = result.append(res, ignore_index=True)
    result_final = result_final.append(result)#, ignore_index=True)
    
print(result_final)
result_final = result_final.loc[lambda x : x>0]


# In[36]:


# Plot
ax = resultTX.plot.kde()
plt.title("Distribution of relatedness (Texas, 2016-2020)", fontsize = 12)
plt.xlabel('Relatedness', fontsize=10)
plt.ylabel('Percentage of new green products', fontsize=10)
plt.show()

ax = resultFL.plot.kde()
plt.title("Distribution of relatedness (Florida, 2016-2020)", fontsize = 12)
plt.xlabel('Relatedness', fontsize=10)
plt.ylabel('Percentage of new green products', fontsize=10)
plt.show()

ax = result_final.plot.kde()
plt.title("Distribution of relatedness (US, 2016-2020)", fontsize = 12)
plt.xlabel('Relatedness', fontsize=10)
plt.ylabel('Percentage of new green products', fontsize=10)
plt.show()


# In[37]:


ax = resultTX.plot.kde(label='TX')
ax = resultFL.plot.kde(label='FL')
ax = result_final.plot.kde(label='US')
plt.title("Distribution of relatedness (2016-2020)", fontsize = 12)
plt.xlabel('Relatedness', fontsize=10)
plt.ylabel('Percentage of new green products', fontsize=10)
plt.legend()
plt.show()


# ## 3.	How have these states evolved over the years in their green economy?

# In[38]:


print('\033[1m' ,"Number of green products with RCA > 1")
print(" Year\tTexas\tchange\t %change_TX\tFlorida\t  change  %change_FL")
cntTX = 0
cntFL = 0

for year in years:
        RCA_1_TX = RCA_df[year][(RCA_df[year].State == 'Texas')  & (RCA_df[year].RCA >= 1)]
        changeTX = len(RCA_1_TX) - cntTX
        cntTX = len(RCA_1_TX)
        percentTX = changeTX/cntTX * 100
        
        RCA_1_FL = RCA_df[year][(RCA_df[year].State == 'Florida')  & (RCA_df[year].RCA >= 1)]
        changeFL = len(RCA_1_FL) - cntFL
        cntFL = len(RCA_1_FL)
        percentFL = changeFL/cntFL * 100
          
        if year == 2016:
            continue

        print('\033[0m', year,'\t', len(RCA_1_TX), '%10s' % changeTX, '%10s' % ("%.2f" % percentTX),                              '\t', len(RCA_1_FL), '%10s' % changeFL, '%10s' % ("%.2f" % percentFL))


# ## Network based on green product category

# In[39]:


graph = gps_network_graph[2020]

# convert nodes list from graph to df
nodes_list = pd.DataFrame(list(graph.nodes())).reset_index()
nodes_list.columns = ["order", "HSCode"]

# combine green products df and nodes
nodes_category = green_prod.merge(nodes_list, how = "outer", on = "HSCode")

nodes_category = nodes_category.sort_values(by = "order")
nodes_category = nodes_category[nodes_category['order'].notna()]

nodes_category = nodes_category.merge(gp_category, how = "inner", on = "Medium")

nodes_category.groupby(['Medium','Theme']).size().to_frame('Count').reset_index().sort_values(by = "Count", ascending = False)
   # with pd.option_context('display.max_colwidth', None):
#    display(nodes_category.groupby(['Medium','Theme']).size().reset_index())



# In[40]:


# plot
fig = plt.figure(figsize = (50,50))
ax = fig.gca()

nodePos = gps_node_pos[2020]

cat_legend = [Line2D([0], [0], marker='o', color='w', label='APC', markerfacecolor='cyan', markersize=25),
              Line2D([0], [0], marker='o', color='w', label='CRE', markerfacecolor='yellow', markersize=25),
              Line2D([0], [0], marker='o', color='w', label='EPP', markerfacecolor='brown', markersize=25),
              Line2D([0], [0], marker='o', color='w', label='HEM', markerfacecolor='purple', markersize=25),
              Line2D([0], [0], marker='o', color='w', label='MON', markerfacecolor='green', markersize=25),
              Line2D([0], [0], marker='o', color='w', label='NRP', markerfacecolor='magenta', markersize=25),
              Line2D([0], [0], marker='o', color='w', label='NVA', markerfacecolor='indigo', markersize=25),
              Line2D([0], [0], marker='o', color='w', label='REP', markerfacecolor='red', markersize=25),
              Line2D([0], [0], marker='o', color='w', label='SWM', markerfacecolor='orange', markersize=25),
              Line2D([0], [0], marker='o', color='w', label='SWR', markerfacecolor='grey', markersize=25),
              Line2D([0], [0], marker='o', color='w', label='WAT', markerfacecolor='blue', markersize=25)]

ax.legend(cat_legend, ['APC', 'CRE', 'EPP', 'HEM', 'MON', 'NRP', 'NVA', 'REP', 'SWM', 'SWR','WAT'],fontsize = 30)    


plt.title("Product Space Network - Category (2020)", fontsize = 60, y=1.0, pad=-70)
nx.draw_networkx(graph, pos = nodePos, node_color = nodes_category.Color.values, with_labels = False, label = nodes_category.Medium)#,  node_size=200, width=1, )


# ## Node size based on export value of all green products (2020)

# In[41]:


for state in ['Texas', 'Florida']:
    gp_state_year = RCA_df[2020][RCA_df[2020].State == state]
    pos = gps_node_pos[2020]
    graph = gps_network_graph[2020]

    gp_state_year['Node_size'] = gp_state_year['state export']/100000

    nodes_list = pd.DataFrame(list(graph.nodes())).reset_index()
    nodes_list.columns = ["order", "HSCode"]

    # combine RCA df and nodes
    nodes_state_yr = gp_state_year.merge(nodes_list, how = "outer", on = "HSCode")
    nodes_state_yr = nodes_state_yr.sort_values(by = "order")

    nodes_state_yr = nodes_state_yr[nodes_state_yr['order'].notna()]

    # assign size for nodes not in list
    nodes_state_yr.loc[nodes_state_yr.Node_size.isnull(), 'Node_size'] = 10

    nodes_state_yr = nodes_state_yr.merge(gp_category, how = "outer", on = "Medium")
  #  print(gp_state_year)
    high_export = gp_state_year.nlargest(10, 'state export')[['HSCode','Medium','state export']].reset_index(drop='True')
    print(state)
    print(high_export)
    # plot
    fig = plt.figure(figsize = (50,50))
    ax = fig.gca()

    ax.set_title("GPS node size based on exports - " + state , fontsize = 60, y=1.0, pad=-70)

    ax.legend(cat_legend, ['APC', 'CRE', 'EPP', 'HEM', 'MON', 'NRP', 'NVA', 'REP', 'SWM', 'SWR','WAT'],fontsize = 30)    

    nx.draw(graph, pos=pos, width=1, with_labels = False, node_size = nodes_state_yr.Node_size.values, node_color = nodes_category.Color.values)


# ## Filter on category (2020)

# In[42]:


# choose category 
category = "REP"


# In[43]:


def filter_node(n1):
    if n1 in list(nodes_category1.HSCode):
        return True
    else:
        return False


# In[44]:


# filter nodes
nodes_category1 = nodes_category[nodes_category.Medium == category] 
# plot
fig = plt.figure(figsize = (50,50))
ax = fig.gca()

nodePos = gps_node_pos[2020]
#nodePos=nx.kamada_kawai_layout(graph)

cat_legend1 = [Line2D([0], [0], marker='o', color = 'w', markerfacecolor=nodes_category1.Color.values[1], markersize=25)]

ax.legend(cat_legend1, nodes_category1.Medium, fontsize = 40)

view = nx.subgraph_view(graph, filter_node = filter_node)

plt.title("Product Space Network - Category (" + str(nodes_category1.Medium.values[1]) + ")", fontsize = 60, y=1.0, pad=-70)

nx.draw_networkx(view, pos = nodePos, node_size=5000, width=1, node_color = nodes_category1.Color.values, with_labels = True, label = nodes_category.Medium)


# ## Filter on RCA > 1 , node size with export values, color based on category
# ### Texas

# In[45]:


def filter_node_txrca(n1):
    if n1 in list(gp_tx.HSCode):
        return True
    else:
        return False


# In[46]:


for year in years:
    #year = 2020
    state = 'Texas'
    graph = gps_network_graph[year]
    nodePos = gps_node_pos[year]

    
    # select the RCA df for a year and filter by state
    gp_tx = RCA_df[year][RCA_df[year].State == state]
    
    gp_tx = gp_tx.loc[(gp_tx.RCA) >= 1]
    
    gp_tx['Node_size'] = gp_tx['state export']/100000
    
    gp_tx = gp_tx.merge(gp_category, how = "inner", on = "Medium")

    # plot
    fig = plt.figure(figsize = (50,50))
    ax = fig.gca()
    ax.text( 0.99, 0.97,
        "GP with RCA>=1 is " + str(len(gp_tx)),
        style='italic',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=50)
        
    view = nx.subgraph_view(graph, filter_node = filter_node_txrca)
    
    nodes_list = pd.DataFrame(list(view.nodes())).reset_index()
    nodes_list.columns = ["order", "HSCode"]
    
    gp_tx = gp_tx.merge(nodes_list, how = "outer", on = "HSCode")
    gp_tx = gp_tx.sort_values(by = "order")
    
    gp_tx = gp_tx[gp_tx['order'].notna()]

    plt.title("Green Product Space - " + state+ " " +str(year) , fontsize = 60, y=1.0, pad=-70)
        
    ax.legend(cat_legend, ['APC', 'CRE', 'EPP', 'HEM', 'MON', 'NRP', 'NVA', 'REP', 'SWM', 'SWR','WAT'],fontsize = 30,bbox_to_anchor=[0.98, 0.95])    
    
    nx.draw(view, pos=nodePos,  width=1, with_labels = True, node_size = list(gp_tx.Node_size.values), node_color = gp_tx.Color.values)


# In[47]:


# RCA > 1
gp_tx_exports = gp_tx[['HSCode','state export','Commodity', 'Medium', 'Theme', 'Color' ,'Node_size']].sort_values(by=['state export'], ascending=False)
print(list(gp_tx_exports.Medium.unique()))
gp_tx_exports.head()


# In[48]:


# All RCA
gp_tx = RCA_df[year][RCA_df[year].State == 'Texas']
gp_tx_exports = gp_tx[['HSCode','state export','Commodity', 'Medium','RCA']].sort_values(by=['state export'], ascending=False)
gp_tx_exports


# ## Filter on RCA > 1 , node size with export values, color based on category
# ### Florida

# In[49]:


def filter_node_flrca(n1):
    if n1 in list(gp_fl.HSCode):
        return True
    else:
        return False


# In[50]:


state = 'Florida'
year = 2020
graph = gps_network_graph[year]
nodePos = gps_node_pos[year]
# select the RCA df for a year and filter by state
gp_fl = RCA_df[year][RCA_df[year].State == state]

gp_fl = gp_fl.loc[(gp_fl.RCA) >= 1]

gp_fl['Node_size'] = gp_fl['state export']/100000

gp_fl = gp_fl.merge(gp_category, how = "inner", on = "Medium")

# plot
fig = plt.figure(figsize = (50,50))
ax = fig.gca()
ax.text( 0.98, 0.98,
    "GP with RCA>=1 is " + str(len(gp_fl)),
    style='italic',
    verticalalignment='center', horizontalalignment='right',
    transform=ax.transAxes,
    color='green', fontsize=50)
    
view = nx.subgraph_view(graph, filter_node = filter_node_flrca)

nodes_list = pd.DataFrame(list(view.nodes())).reset_index()
nodes_list.columns = ["order", "HSCode"]

# combine RCA df and nodes
gp_fl = gp_fl.merge(nodes_list, how = "inner", on = "HSCode")
gp_fl = gp_fl.sort_values(by = "order")

gp_fl = gp_fl[gp_fl['order'].notna()]

plt.title("Green Product Space - " + state+ " " +str(year) , fontsize = 60, y=1.0, pad=-70)
ax.legend(cat_legend, ['APC', 'CRE', 'EPP', 'HEM', 'MON', 'NRP', 'NVA', 'REP', 'SWM', 'SWR','WAT'],fontsize = 30,bbox_to_anchor=[0.98, 0.95])    


nx.draw(view, pos=nodePos,  width=1, with_labels = True, node_size = list(gp_fl.Node_size.values), node_color = gp_fl.Color.values)


# In[51]:


gp_fl_exports = gp_fl[['HSCode','state export','Commodity', 'Medium', 'Theme', 'Color' ,'Node_size']].sort_values(by=['state export'], ascending=False)
gp_fl_exports


# ## Misc

# In[52]:


initial_year = 2017
final_year = 2019


# In[53]:


initial_df = RCA_df[initial_year]
final_df = RCA_df[final_year]
new_gp_list = pd.DataFrame()

states = ['Texas', 'Florida']
# new green products for all states from 2016-2018
for state in states:
    new_green = pd.merge(initial_df[(initial_df.State == state) & (initial_df.RCA < 0.5)],
                           final_df[(final_df.State == state) & (final_df.RCA >=1)], how ='inner', on = ["HSCode",'Commodity', 'State','Medium'])
    new_green = new_green[['State','HSCode','Commodity','Medium', 'Time_x','Time_y','RCA_x','RCA_y']]
    new_gp_list = new_gp_list.append(new_green)


# In[54]:


new_gp_list


# In[55]:


new_gp_TX = new_gp_list[new_gp_list.State == 'Texas']
RCA1TX = MCP_matrix[2016][MCP_matrix[2016].index == 'Texas']
RCA1TX = RCA1TX.loc[:, (RCA1TX != 0).any(axis=0)]
RCA1TX


# In[56]:


relatedness_TX = proximity_dict[2016]
relatedness_TX = relatedness_TX.filter(items = new_gp_TX.HSCode, axis = 0)
relatedness_TX = relatedness_TX.filter(items = RCA1TX.columns)
relatedness_TX = relatedness_TX.fillna(0)
relatedness_TX


# In[57]:


relatedness_TX = relatedness_TX.max(axis = 1)


# In[58]:


df = pd.DataFrame(relatedness_TX, columns = ['Max proximity'])


# In[59]:


new_gp_TX


# In[60]:


def filter_node_newgp(n1):
    if n1 in list(new_gp_TX.HSCode):
        return True
    else:
        return False


# In[61]:


g1 = gps_network_graph[2017]
p1 = gps_node_pos[2017]

g1_view = nx.subgraph_view(g1, filter_node = filter_node_newgp)


g2 = gps_network_graph[2019]
p2 = gps_node_pos[2019]
g2_view = nx.subgraph_view(g2, filter_node = filter_node_newgp)


# In[62]:


nx.draw(g1_view, pos=p1,  width=1, with_labels = True)# node_size = list(gp_fl.Node_size.values), node_color = gp_fl.Color.values)


# In[63]:


nx.draw(g2_view, pos=p2,  width=1, with_labels = True)# node_size = list(gp_fl.Node_size.values), node_color = gp_fl.Color.values)


# ## New green products TX (2017 - 2019)

# In[64]:


year = 2019
state = 'Texas'
graph = gps_network_graph[year]
nodePos = gps_node_pos[year]

# select the RCA df for a year and filter by state
gp_tx = RCA_df[year][RCA_df[year].State == state]

gp_tx = gp_tx.loc[(gp_tx.RCA) >= 1]

   # gp_tx['Node_size'] = gp_tx['state export']/100000

gp_tx = gp_tx.merge(gp_category, how = "inner", on = "Medium")

#‘so^>v<dph8’.
gp_tx['Node_shape'] = 'o'
gp_tx['Node_size'] = 100
gp_tx['label_size'] = ''

for item in list(new_gp_TX.HSCode):
    if item in list(gp_tx.HSCode):
        gp_tx.loc[gp_tx.HSCode == item, 'Node_shape']  = 'h'  
        gp_tx.loc[gp_tx.HSCode == item, 'Node_size']  = 30000 
        gp_tx.loc[gp_tx.HSCode == item, 'label_size'] = item


#for item in list(new_gp_TX.HSCode):
 #   if item in list(gp_tx.HSCode):
  #      gp_tx.loc[gp_tx.HSCode == item, 'Node_size']  = 10000 

view = nx.subgraph_view(graph, filter_node = filter_node_txrca)

nodes_list = pd.DataFrame(list(view.nodes())).reset_index()
nodes_list.columns = ["order", "HSCode"]

gp_tx = gp_tx.merge(nodes_list, how = "outer", on = "HSCode")
gp_tx = gp_tx.sort_values(by = "order")

gp_tx = gp_tx[gp_tx['order'].notna()]

node_list = list(view.nodes())

color_dict = { k:v for k,v in zip(node_list, gp_tx.Color)}
nx.set_node_attributes(view, values = color_dict, name='color')

size_dict = { k:v for k,v in zip(node_list, gp_tx.Node_size)}
nx.set_node_attributes(view, values = size_dict, name='size')

label_dict = { k:v for k,v in zip(node_list, gp_tx.label_size)}
nx.set_node_attributes(view, values = label_dict, name='label')

node_shapes = list(gp_tx.Node_shape)


# In[65]:


for i,node in enumerate(view.nodes()):
    view.nodes[node]['shape'] = node_shapes[i]


# In[66]:


fig = plt.figure(figsize = (50,50))
ax = fig.gca()
#nx.draw(view, pos=nodePos, width=1, with_labels = True)#, node_size = list(gp_tx.Node_size.values), node_color = gp_tx.Color.values)#, node_shape = list(gp_tx.Node_shape.values))

nx.draw_networkx_edges(view,nodePos) # draw edges
nx.draw_networkx_labels(view,nodePos, labels=label_dict, font_size = 40, font_color = 'gray')


for shape in set(node_shapes):
    # the nodes with the desired shapes
    node_list = [node for node in view.nodes() if view.nodes[node]['shape'] == shape]
    nx.draw_networkx_nodes(view,nodePos,
                           nodelist = node_list,
                           node_size = [view.nodes[node]['size'] for node in node_list],
                           node_color= [view.nodes[node]['color'] for node in node_list],
                           node_shape = shape)


# ## Categories in network 

# In[67]:


for category in list(gp_category.Medium):
    graph = gps_network_graph[2020]

    # convert nodes list from graph to df
    nodes_list = pd.DataFrame(list(graph.nodes())).reset_index()
    nodes_list.columns = ["order", "HSCode"]
    
    # combine green products df and nodes
    nodes_category = green_prod.merge(nodes_list, how = "outer", on = "HSCode")
    
    nodes_category = nodes_category.sort_values(by = "order")
    nodes_category = nodes_category[nodes_category['order'].notna()]

    nodes_category = nodes_category.merge(gp_category, how = "inner", on = "Medium")
    nodes_category.groupby(['Medium','Color']).size().reset_index()
    
        # plot
    fig = plt.figure(figsize = (50,50))
    ax = fig.gca()
    
    nodePos = gps_node_pos[2020]
    
    nodes_category['Node_shape'] = 'o'
    nodes_category['Node_size'] = 100
    

    for item in list(nodes_category.HSCode):
        if item in list(nodes_category[nodes_category.Medium == category]['HSCode']):
            nodes_category.loc[nodes_category.HSCode == item, 'Node_shape']  = 'v'
            nodes_category.loc[nodes_category.HSCode == item, 'Node_size']  = 7000
            
    ax.legend(cat_legend, ['APC', 'CRE', 'EPP', 'HEM', 'MON', 'NRP', 'NVA', 'REP', 'SWM', 'SWR','WAT'],fontsize = 30)    

    node_shapes = list(nodes_category.Node_shape)
    
    node_list = list(graph.nodes())
    color_dict = { k:v for k,v in zip(node_list, nodes_category.Color)}
    nx.set_node_attributes(graph, values = color_dict, name='color')
    
    size_dict = { k:v for k,v in zip(node_list,nodes_category.Node_size)}
    nx.set_node_attributes(graph, values = size_dict, name='size')
    
    for i,node in enumerate(graph.nodes()):
        graph.nodes[node]['shape'] = node_shapes[i]
    
    plt.title("Product Space Network - Category " + category, fontsize = 60, y=1.0, pad=-70)
    
    nx.draw_networkx_edges(graph,nodePos) # draw edges
    #nx.draw_networkx_labels(graph,nodePos) # draw node labels

    for shape in set(node_shapes):
        # the nodes with the desired shapes
        node_list = [node for node in graph.nodes() if graph.nodes[node]['shape'] == shape]
        nx.draw_networkx_nodes(view,nodePos,
                           nodelist = node_list,
                           node_size = [graph.nodes[node]['size'] for node in node_list],
                           node_color= [graph.nodes[node]['color'] for node in node_list],
                           node_shape = shape)


# In[68]:


list(gp_category.Medium)


# In[69]:


set(node_shapes)


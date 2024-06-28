
import streamlit as st
import pandas as pd
from st_aggrid import AgGrid

house = pd.read_csv('house_clean.csv')

def main():
    st.write('Minimal Example')
    st.header('Halaman Streamlit Akmal')
    st.subheader('This is SubHeader')
    st.markdown('# Rendering Markdown ')
    st.write('Some Phytagorean Equation : ')
    st.latex('C^2 = a^2+B^2')
    
    st.dataframe(house)

    st.write('Metrics')
    st.metric(label="Temperature", value="70 °F", delta="-1.2 °F")
    st.write('Menampilkan Dataframe dengan St AgGrid')
    AgGrid(house)
    
    st.table([x for x in range(1,5)])

if __name__ == '__main__':
    main()
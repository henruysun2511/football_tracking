import pickle

def check_stubs():
    try:
        with open('stubs/tracks_full.pkl', 'rb') as f:
            tracks = pickle.load(f)
            
        print("Checking tracks_full.pkl...")
        players = tracks['players']
        print(f"Total frames: {len(players)}")
        
        # Check first few frames for player 1
        tid_to_check = None
        for tid in players[0]:
            tid_to_check = tid
            break
            
        if tid_to_check is not None:
            print(f"Checking player {tid_to_check}...")
            for i in range(0, min(30, len(players))):
                data = players[i].get(tid_to_check)
                if data:
                    print(f"Frame {i}: pos={data.get('position')} pos_adj={data.get('position_adjusted')} pos_tx={data.get('position_transformed')} speed={data.get('speed')} dist={data.get('distance')}")
                    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    check_stubs()

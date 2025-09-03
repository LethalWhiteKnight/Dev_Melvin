import streamlit as st
import time
from spell_checker import SpellChecker, DEMO_CORPUS, levenshtein_matrix, backtrace_edits

# Initialize spell checker
sc = SpellChecker(DEMO_CORPUS)

st.set_page_config(page_title="Levenshtein Spell Checker", page_icon="📝")

st.title("Spell Checker (Levenshtein Distance)")
st.write("Type a word below, and see suggestions + animated backtrace of edits.")

# Input box
word = st.text_input("Enter a word:", "")

if word:
    best = sc.correct(word)
    suggestions = sc.suggest(word, k=5)
    candidates = sc.candidates(word)

    st.subheader("Best Suggestion")
    st.success(best if best else "No suggestion found.")

    st.subheader("Debug Info")
    st.write(f"**Input word:** {word}")
    st.write(f"**Total candidates found:** {len(candidates)}")
    if candidates:
        st.write(f"**Candidates:** {', '.join(sorted(candidates))}")
    else:
        st.write("**No candidates found in vocabulary**")

    st.subheader("Top Candidates")
    if suggestions:
        for cand, dist, freq in suggestions:
            st.write(f"✅ **{cand}**  | Distance = {dist}, Frequency = {freq}")
    else:
        st.warning("No candidates found in vocabulary.")

    # Show animated backtrace for best correction
    if best and best != word:
        st.subheader("Backtrace Animation")
        
        # Prepare animation data
        dp = levenshtein_matrix(word, best)
        edits = backtrace_edits(word, best, dp)
        
        # Display all steps automatically with proper simulation
        # Create a proper forward simulation of the transformation
        current_word = list(word)
        
        # Track positions in both words
        i, j = 0, 0  # positions in current_word and target_word
        steps = []
        
        for op, x, y in edits:
            if op == "MATCH":
                steps.append((f"MATCH: `{x}` stays as `{y}`", None))
                i += 1
                j += 1
            elif op == "SUB":
                steps.append((f"SUBSTITUTE: `{x}` → `{y}`", (i, y, "sub")))
                i += 1
                j += 1
            elif op == "DEL":
                steps.append((f"DELETE: `{x}`", (i, None, "del")))
                i += 1
            elif op == "INS":
                steps.append((f"INSERT: `{y}`", (i, y, "ins")))
                j += 1
        
        # Now simulate the steps
        for step, (msg, change) in enumerate(steps, 1):
            if change is not None:
                pos, new_char, op_type = change
                if op_type == "del":
                    # Delete character at position
                    if pos < len(current_word):
                        current_word.pop(pos)
                elif op_type == "ins":
                    # Insert character at position
                    current_word.insert(pos, new_char)
                elif op_type == "sub":
                    # Substitute character at position
                    if pos < len(current_word):
                        current_word[pos] = new_char
            
            word_str = "".join(current_word)
            st.markdown(f"### Step {step}: {msg}")
            st.markdown(f"**Current Word →** `{word_str}`")
            st.markdown("---")

        st.success(f" Final corrected word: **{best}**") 
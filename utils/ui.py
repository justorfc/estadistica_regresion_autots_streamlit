from __future__ import annotations

import streamlit as st


def page_title(title: str, subtitle: str | None = None) -> None:
    """Muestra un encabezado uniforme para todas las páginas."""
    st.title(title)
    if subtitle:
        st.markdown(f"**{subtitle}**")
    st.divider()


def info_box(title: str, body: str) -> None:
    """Crea un bloque visual sencillo para orientar al usuario."""
    st.markdown(
        f"""
        <div style="padding: 1rem; border-radius: 0.75rem; background: #F5F7FA;
                    border: 1px solid #E5E7EB; margin-bottom: 1rem;">
            <strong>{title}</strong><br>
            <span>{body}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_model_note() -> None:
    """Nota metodológica común para modelos estadísticos."""
    st.caption(
        "Nota: los resultados dependen de la calidad de los datos, del cumplimiento "
        "razonable de los supuestos del modelo y de la interpretación técnica del problema."
    )

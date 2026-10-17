"""Testes do tema (app/theme.py + app/assets/theme.css).

O AppTest não renderiza CSS, então aqui o CSS é lido como texto. O ponto
principal é uma regressão real: o CSS global escondia o `stToolbar` inteiro e,
nas versões recentes do Streamlit, é dentro dele que mora o botão de REABRIR a
barra lateral — depois de fechada, ela não voltava mais.
"""

import re
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import theme  # noqa: E402

# Controles que levam o usuário de volta à barra lateral (nomes de versões
# recentes e antigas do Streamlit) e o contêiner que guarda um deles.
SIDEBAR_CONTROLS = ("stExpandSidebarButton", "stSidebarCollapsedControl", "stSidebarCollapseButton")
CONTAINERS_OF_THE_CONTROL = ("stToolbar", "stHeader")


def _rules(css: str) -> list[tuple[str, str]]:
    """(seletores, declarações) de cada regra simples do CSS (sem comentários
    nem a linha de @import, que não são regras).
    """
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    css = re.sub(r"@import[^;]*;", "", css)
    return [(sel.strip(), body) for sel, body in re.findall(r"([^{}@]+)\{([^{}]*)\}", css)]


@pytest.fixture(scope="module")
def css() -> str:
    return theme.build_css()


def _hiding_selectors(css: str) -> list[str]:
    hidden = []
    for selectors, body in _rules(css):
        if re.search(r"display\s*:\s*none|visibility\s*:\s*hidden", body):
            hidden += [s.strip() for s in selectors.split(",")]
    return hidden


def test_no_rule_hides_the_sidebar_controls_or_the_elements_that_hold_them(css):
    for selector in _hiding_selectors(css):
        for name in SIDEBAR_CONTROLS + CONTAINERS_OF_THE_CONTROL:
            # `stToolbarActions` (só as ações) pode sumir; `stToolbar` sozinho, não
            assert f'"{name}"' not in selector, f"o CSS esconde {name}: {selector}"


def test_streamlit_chrome_is_hidden_but_only_the_parts_that_are_not_the_control(css):
    hidden = " ".join(_hiding_selectors(css))
    for name in ("stToolbarActions", "stAppDeployButton", "stMainMenu", "stStatusWidget", "stDecoration"):
        assert name in hidden, name
    assert "footer" in hidden


def test_expand_control_stays_clickable_even_with_a_click_through_header(css):
    rules = _rules(css)
    header = next(body for sel, body in rules if sel == '[data-testid="stHeader"]')
    assert "pointer-events: none" in header  # o cabeçalho transparente não captura cliques...
    clickable = " ".join(sel for sel, body in rules if "pointer-events: auto" in body)
    assert "stExpandSidebarButton" in clickable and "stSidebarCollapsedControl" in clickable  # ...mas o controle sim


def test_collapse_button_is_always_visible_for_touch_and_keyboard(css):
    rule = next(body for sel, body in _rules(css) if sel == '[data-testid="stSidebarCollapseButton"]')
    assert "visibility: visible" in rule


def test_css_variables_come_from_the_tokens(css):
    root = css[css.index(":root") : css.index("}", css.index(":root"))]
    for token, value in (("bg", theme.BG), ("surface", theme.SURFACE), ("accent", theme.ACCENT), ("risk-high", theme.RISK_HIGH)):
        assert f"--{token}: {value};" in root


def test_font_import_comes_first_and_has_a_fallback_stack(css):
    assert css.startswith("@import url(")  # @import só vale antes de qualquer regra
    assert "-apple-system" in theme.FONT_STACK and "sans-serif" in theme.FONT_STACK  # offline cai no sistema


def test_legacy_names_still_resolve_for_the_scientific_section():
    assert theme.ELECTRIC_BLUE == theme.ACCENT and theme.CORAL == theme.RISK_HIGH and theme.MUTED == theme.TEXT_SECONDARY
    assert set(theme.CLASS_COLORS) == {"Legítima", "Fraude"}

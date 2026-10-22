"""Asistente virtual de Cálculo y Precálculo (Streamlit + SymPy)."""
import random
    
import numpy as np
import pandas as pd
import streamlit as st
import sympy as sp
from sympy import S, diff, integrate, latex, limit, solveset, symbols
from sympy.calculus.util import continuous_domain, function_range
from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application,
                                        parse_expr, standard_transformations)

# ====== DATOS DEL EQUIPO: cámbialos ======
NOMBRE_APP = "Solvemath"
INTEGRANTES = "Victor Daniel"
CURSO = "Precálculo"
SECCION = "Sección A"

X = symbols("x")
TRANSF = standard_transformations + (implicit_multiplication_application, convert_xor)
LOCAL = {"x": X, "pi": sp.pi, "e": sp.E, "ln": sp.log}


# ---------- Utilidades ----------
def leer(texto):
    """Convierte el texto del usuario en expresión SymPy o lanza ValueError con mensaje claro."""
    if not texto or not texto.strip():
        raise ValueError("Falta un dato: la caja está vacía.")
    try:
        e = parse_expr(texto, local_dict=LOCAL, transformations=TRANSF)
    except Exception:
        raise ValueError("No pude entender la expresión. Usa x como variable, ** o ^ para "
                         "potencias y * para multiplicar. Ejemplo: 3*x**2 + 2*x - 1")
    if not isinstance(e, sp.Basic):
        raise ValueError("Eso no es una expresión matemática válida.")
    extra = e.free_symbols - {X}
    if extra:
        raise ValueError("Solo puedo usar la variable x; encontré: " + ", ".join(map(str, extra)))
    return e


def leer_punto(texto):
    """Como leer(), pero acepta infinito (oo, inf, -oo)."""
    t = (texto or "").strip().lower().replace(" ", "")
    if t in ("oo", "inf", "infinito", "+oo", "∞"):
        return sp.oo
    if t in ("-oo", "-inf", "-infinito", "-∞"):
        return -sp.oo
    return leer(texto)


def tex(e):
    st.latex(latex(e))


def seguro(fn, *args):
    """Ejecuta fn y convierte los errores en mensajes amigables."""
    try:
        fn(*args)
    except ValueError as err:
        st.error(f"⚠️ {err}")
    except Exception as err:
        st.warning(f"No pude resolver este caso ({type(err).__name__}). "
                   "Revisa la expresión o prueba con una más simple.")


# ---------- Módulos ----------
def algebra(op, t1, t2=""):
    if op == "Resolver ecuación":
        izq, der = t1.split("=", 1) if "=" in t1 else (t1, "0")
        eq = leer(izq) - leer(der)
        st.markdown("**Paso 1: pasar todo a un lado**")
        tex(sp.Eq(sp.expand(eq), 0))
        try:
            a, b, c = sp.Poly(eq, X).all_coeffs() if sp.Poly(eq, X).degree() == 2 else (None,) * 3
        except Exception:
            a = None
        if a is not None:
            st.markdown("**Fórmula general:** $x=\\dfrac{-b\\pm\\sqrt{b^2-4ac}}{2a}$")
            st.latex(f"a={latex(a)},\\ b={latex(b)},\\ c={latex(c)}\\ \\Rightarrow\\ "
                     f"\\Delta=b^2-4ac={latex(b**2 - 4*a*c)}")
        st.markdown("**Soluciones reales:**")
        tex(solveset(eq, X, S.Reals))
    elif op == "Factorizar":
        f = leer(t1)
        st.markdown("**Expresión expandida y factorizada:**")
        st.latex(latex(sp.expand(f)) + " = " + latex(sp.factor(f)))
    elif op == "Dominio y rango":
        f = leer(t1)
        st.markdown("**Dominio** (valores de x permitidos):")
        tex(continuous_domain(f, X, S.Reals))
        st.markdown("**Rango** (valores que toma f):")
        tex(function_range(f, X, S.Reals))
    else:
        f, g = leer(t1), leer(t2)
        st.markdown("**Composición:** se sustituye una función dentro de la otra.")
        st.latex("(f\\circ g)(x)=f(g(x))=" + latex(sp.simplify(f.subs(X, g))))
        st.latex("(g\\circ f)(x)=g(f(x))=" + latex(sp.simplify(g.subs(X, f))))


def grafica(texto, xmin, xmax):
    f = leer(texto)
    if xmin >= xmax:
        raise ValueError("El mínimo del intervalo debe ser menor que el máximo.")
    xs = np.linspace(xmin, xmax, 400)
    fn = sp.lambdify(X, f, "numpy")
    with np.errstate(all="ignore"):
        ys = np.array(fn(xs), dtype=float) * np.ones_like(xs)
    ys[np.abs(ys) > 1e6] = np.nan  # corta las asíntotas
    st.line_chart(pd.DataFrame({"f(x)": ys}, index=xs))
    f1 = diff(f, X)
    st.markdown("**Interpretación:**")
    try:
        st.latex("\\text{Corte con eje }y:\\ f(0)=" + latex(f.subs(X, 0)))
    except Exception:
        st.write("No hay corte con el eje y (f no está definida en 0).")
    st.markdown("Cortes con el eje x (f(x)=0):")
    tex(solveset(f, X, S.Reals))
    st.markdown("Puntos críticos (f'(x)=0):")
    tex(solveset(f1, X, S.Reals))
    try:
        st.markdown("Crece donde f'(x) > 0:")
        tex(solveset(f1 > 0, X, S.Reals))
        st.markdown("Decrece donde f'(x) < 0:")
        tex(solveset(f1 < 0, X, S.Reals))
    except Exception:
        st.write("No pude determinar los intervalos de crecimiento para esta función.")


def limites(texto, punto, lado="ambos"):
    f, p = leer(texto), leer_punto(punto)
    st.markdown("**Paso 1: sustitución directa**")
    try:
        directo = sp.sympify(f.subs(X, p))
    except Exception:
        directo = sp.nan
    st.latex("f(" + latex(p) + ")=" + latex(directo))
    if directo.is_finite:
        st.write("Da un número: ese es el límite (la función es continua en el punto).")
    else:
        st.write("Sale una forma indeterminada o indefinida: hay que simplificar.")
        st.markdown("**Paso 2: factorizar y simplificar**")
        tex(sp.simplify(sp.factor(f)))
    st.markdown("**Respuesta:**")
    if p.is_infinite:
        st.latex("\\lim_{x\\to " + latex(p) + "} f(x)=" + latex(limit(f, X, p)))
    elif lado == "ambos":
        izq, der = limit(f, X, p, "-"), limit(f, X, p, "+")
        st.latex("\\lim_{x\\to p^-}=" + latex(izq) + ",\\quad \\lim_{x\\to p^+}=" + latex(der))
        if izq == der:
            st.latex("\\lim_{x\\to " + latex(p) + "} f(x)=" + latex(izq))
        else:
            st.write("Los límites laterales son distintos: **el límite no existe**.")
    else:
        d = "-" if lado == "izquierda" else "+"
        st.latex("\\lim_{x\\to " + latex(p) + "^{" + d + "}} f(x)=" + latex(limit(f, X, p, d)))


def derivada(texto):
    f = leer(texto)
    st.markdown("**Fórmulas:** potencia $(x^n)'=nx^{n-1}$, constante $(c)'=0$, suma $(u+v)'=u'+v'$.")
    st.markdown("**Derivada de cada término:**")
    for t in sp.Add.make_args(f):
        st.latex("\\frac{d}{dx}\\left(" + latex(t) + "\\right)=" + latex(diff(t, X)))
    f1 = diff(f, X)
    st.markdown("**Respuesta:**")
    st.latex("f'(x)=" + latex(f1))
    try:  # verificación numérica con el cociente incremental
        x0, h = 1.5, 1e-6
        aprox = (float(f.subs(X, x0 + h)) - float(f.subs(X, x0))) / h
        exacto = float(f1.subs(X, x0))
        ok = abs(aprox - exacto) < 1e-3 * max(1, abs(exacto))
        st.caption(f"Verificación en x={x0}: (f(x+h)-f(x))/h ≈ {aprox:.4f} y f'({x0}) = {exacto:.4f} "
                   f"→ {'✔ coincide' if ok else '✖ no coincide'}")
    except Exception:
        pass


def integral(texto, a="", b=""):
    f = leer(texto)
    st.markdown("**Fórmulas:** $\\int x^n dx=\\frac{x^{n+1}}{n+1}$ (n ≠ −1), $\\int c\\,dx=cx$.")
    for t in sp.Add.make_args(f):
        st.latex("\\int " + latex(t) + "\\,dx=" + latex(integrate(t, X)))
    F = integrate(f, X)
    if not a.strip() and not b.strip():
        st.markdown("**Respuesta (integral indefinida):**")
        st.latex(latex(F) + "+C")
        return
    if not (a.strip() and b.strip()):
        raise ValueError("Para una integral definida escribe ambos límites, a y b.")
    pa, pb = leer(a), leer(b)
    if pa.free_symbols or pb.free_symbols:
        raise ValueError("Los límites de integración deben ser números.")
    st.markdown("**Regla de Barrow:** $\\int_a^b f\\,dx=F(b)-F(a)$")
    st.latex("F(" + latex(pb) + ")-F(" + latex(pa) + ")=" + latex(F.subs(X, pb))
             + "-(" + latex(F.subs(X, pa)) + ")")
    st.markdown("**Respuesta:**")
    st.latex("=" + latex(sp.simplify(F.subs(X, pb) - F.subs(X, pa))))


def extremos(texto):
    f = leer(texto)
    f1, f2 = diff(f, X), diff(diff(f, X), X)
    st.markdown("**1) Primera derivada**")
    st.latex("f'(x)=" + latex(f1))
    crit = solveset(f1, X, S.Reals)
    st.markdown("**2) Puntos críticos** (resolver f'(x)=0)")
    tex(crit)
    if not isinstance(crit, sp.FiniteSet) or len(crit) == 0:
        st.info("No hay puntos críticos reales finitos que clasificar.")
        return
    st.markdown("**3) Segunda derivada** (criterio: f''>0 mínimo, f''<0 máximo)")
    st.latex("f''(x)=" + latex(f2))
    st.markdown("**4) Clasificación**")
    for p in sorted(crit, key=lambda v: float(v)):
        v2, y = f2.subs(X, p), sp.simplify(f.subs(X, p))
        tipo = "mínimo relativo" if v2 > 0 else "máximo relativo" if v2 < 0 else "no concluyente (f''=0)"
        st.markdown(f"En $x={latex(p)}$: $f''={latex(v2)}$ → **{tipo}**, con $f({latex(p)})={latex(y)}$")


def rectangulo(perimetro):
    P = sp.nsimplify(perimetro)
    if P <= 0:
        raise ValueError("El perímetro debe ser positivo.")
    A = X * (P / 2 - X)
    st.markdown("**Problema:** entre todos los rectángulos de perímetro $P$, ¿cuál tiene área máxima?")
    st.latex("2x+2y=P\\Rightarrow y=" + latex(P / 2 - X) + "\\Rightarrow A(x)=" + latex(sp.expand(A)))
    st.latex("A'(x)=" + latex(diff(A, X)) + "=0\\Rightarrow x=" + latex(P / 4))
    st.latex("A''(x)=" + latex(diff(A, X, 2)) + "<0\\Rightarrow\\text{máximo}")
    st.success(f"Es un cuadrado de lado {P/4} y área máxima {P**2/16}.")


def razon_cambio(r, drdt):
    R, D = sp.nsimplify(r), sp.nsimplify(drdt)
    if R <= 0:
        raise ValueError("El radio debe ser positivo.")
    st.markdown("**Problema:** un círculo crece; ¿qué tan rápido cambia su área?")
    st.latex("A=\\pi r^2\\Rightarrow\\frac{dA}{dt}=2\\pi r\\frac{dr}{dt}")
    st.latex("\\frac{dA}{dt}=2\\pi(" + latex(R) + ")(" + latex(D) + ")=" + latex(2 * sp.pi * R * D))
    st.success(f"dA/dt ≈ {float(2 * sp.pi * R * D):.4f} unidades² por unidad de tiempo.")


# ---------- Práctica con retroalimentación ----------
def generar(tema):
    a, b, c, n = random.randint(2, 6), random.randint(1, 9), random.randint(1, 9), random.randint(2, 4)
    f = a * X**n + b * X + c
    if tema == "Derivadas":
        return dict(tema=tema, f=f, enun="\\frac{d}{dx}\\left(" + latex(f) + "\\right)", tipo="der",
                    pista="Baja el exponente y réstale 1 a cada término; la constante desaparece.")
    if tema == "Integrales":
        return dict(tema=tema, f=f, enun="\\int\\left(" + latex(f) + "\\right)dx", tipo="int",
                    pista="Sube el exponente en 1 y divide entre el nuevo exponente. No escribas +C.")
    k = random.randint(2, 7)
    g = (X**2 - k**2) / (X - k)
    return dict(tema=tema, correcta=2 * k, enun="\\lim_{x\\to " + str(k) + "}" + latex(g), tipo="lim",
                pista="Factoriza el numerador como (x−k)(x+k) y simplifica.")


def practica():
    tema = st.selectbox("Tema", ["Derivadas", "Integrales", "Límites"], key="pra_tema")
    ej = st.session_state.get("ej")
    if st.button("Nuevo ejercicio", key="pra_nuevo") or ej is None or ej["tema"] != tema:
        ej = st.session_state.ej = generar(tema)
        ej["id"] = random.random()
    st.latex(ej["enun"])
    resp = st.text_input("Tu respuesta", key=f"r{ej['id']}")
    if st.button("Comprobar", key="pra_check"):
        try:
            u = leer(resp)
            if ej["tipo"] == "der":
                ok, sol = sp.simplify(u - diff(ej["f"], X)) == 0, diff(ej["f"], X)
            elif ej["tipo"] == "int":
                ok, sol = sp.simplify(diff(u, X) - ej["f"]) == 0, integrate(ej["f"], X)
            else:
                ok, sol = sp.simplify(u - ej["correcta"]) == 0, ej["correcta"]
        except ValueError as err:
            st.error(f"⚠️ {err}")
            return
        if ok:
            st.success("✅ ¡Correcto!")
        else:
            st.error("❌ No es correcta.")
            st.info("Pista: " + ej["pista"])
            st.latex("\\text{Respuesta: }" + latex(sol) + ("+C" if ej["tipo"] == "int" else ""))


# ---------- 10 ejemplos ----------
EJEMPLOS = [
    ("Derivada de un polinomio", derivada, ("x**3 - 6*x**2 + 9*x + 1",)),
    ("Derivada con exponencial", derivada, ("x**2*exp(x)",)),
    ("Integral indefinida", integral, ("3*x**2 + 2*x + 5",)),
    ("Integral definida de 0 a 3", integral, ("x**2", "0", "3")),
    ("Límite con indeterminación 0/0", limites, ("(x**2-9)/(x-3)", "3")),
    ("Límite al infinito", limites, ("(2*x+1)/(x-4)", "oo")),
    ("Resolver ecuación cuadrática", algebra, ("Resolver ecuación", "x**2 - 5*x + 6 = 0")),
    ("Factorizar", algebra, ("Factorizar", "x**3 - x")),
    ("Gráfica de x³ − 3x", grafica, ("x**3 - 3*x", -3, 3)),
    ("Máximos y mínimos", extremos, ("x**4 - 8*x**2",)),
]


# ---------- Interfaz ----------
st.set_page_config(page_title=NOMBRE_APP, page_icon="📐")
st.title(f"📐 {NOMBRE_APP}")
st.caption(f"{INTEGRANTES} · {CURSO} · {SECCION}")
st.info("Usa **x** como variable. Potencia: `**` o `^` · Multiplicar: `*` · "
        "Funciones: sin(x), cos(x), exp(x), ln(x), sqrt(x) · Infinito: oo")

nombres = ["Álgebra", "Gráficas", "Límites", "Derivadas", "Integrales", "Aplicaciones", "Práctica", "Ejemplos"]
t_alg, t_gra, t_lim, t_der, t_int, t_apl, t_pra, t_eje = st.tabs(nombres)

with t_alg:
    op = st.selectbox("¿Qué quieres hacer?", ["Resolver ecuación", "Factorizar", "Dominio y rango",
                                              "Composición de funciones"], key="alg_op")
    t1 = st.text_input("Ecuación o f(x)", "x**2 - 5*x + 6", key="alg_t1")
    t2 = st.text_input("g(x)", "x + 1", key="alg_t2") if op.startswith("Composición") else ""
    if st.button("Resolver", key="alg_btn"):
        seguro(algebra, op, t1, t2)

with t_gra:
    tg = st.text_input("f(x)", "x**3 - 3*x", key="gra_t")
    c1, c2 = st.columns(2)
    xmin = c1.number_input("x mínimo", value=-4.0, key="gra_min")
    xmax = c2.number_input("x máximo", value=4.0, key="gra_max")
    if st.button("Graficar", key="gra_btn"):
        seguro(grafica, tg, xmin, xmax)

with t_lim:
    tl = st.text_input("f(x)", "(x**2 - 9)/(x - 3)", key="lim_t")
    tp = st.text_input("x tiende a (número, oo o -oo)", "3", key="lim_p")
    lado = st.radio("Lado", ["ambos", "izquierda", "derecha"], horizontal=True, key="lim_lado")
    if st.button("Calcular límite", key="lim_btn"):
        seguro(limites, tl, tp, lado)

with t_der:
    td = st.text_input("f(x)", "x**3 - 6*x**2 + 9*x + 1", key="der_t")
    if st.button("Derivar", key="der_btn"):
        seguro(derivada, td)

with t_int:
    ti = st.text_input("f(x)", "3*x**2 + 2*x + 5", key="int_t")
    c1, c2 = st.columns(2)
    ia = c1.text_input("Límite inferior a (vacío = indefinida)", "", key="int_a")
    ib = c2.text_input("Límite superior b", "", key="int_b")
    if st.button("Integrar", key="int_btn"):
        seguro(integral, ti, ia, ib)

with t_apl:
    prob = st.radio("Problema", ["Máximos y mínimos de una función", "Rectángulo de área máxima",
                                 "Razón de cambio (círculo)"], key="apl_op")
    if prob.startswith("Máximos"):
        ta = st.text_input("f(x)", "x**3 - 6*x**2 + 9*x + 1", key="apl_f")
        if st.button("Analizar", key="apl_b1"):
            seguro(extremos, ta)
    elif prob.startswith("Rectángulo"):
        per = st.number_input("Perímetro", value=40.0, key="apl_p")
        if st.button("Resolver", key="apl_b2"):
            seguro(rectangulo, per)
    else:
        c1, c2 = st.columns(2)
        rad = c1.number_input("Radio r", value=5.0, key="apl_r")
        vel = c2.number_input("dr/dt", value=2.0, key="apl_v")
        if st.button("Calcular", key="apl_b3"):
            seguro(razon_cambio, rad, vel)

with t_pra:
    practica()

with t_eje:
    i = st.selectbox("Elige un ejemplo", range(len(EJEMPLOS)), key="ej_sel",
                     format_func=lambda k: f"{k + 1}. {EJEMPLOS[k][0]}")
    st.code(" | ".join(str(a) for a in EJEMPLOS[i][2]))
    if st.button("Resolver ejemplo", key="ej_btn"):
        seguro(EJEMPLOS[i][1], *EJEMPLOS[i][2])
